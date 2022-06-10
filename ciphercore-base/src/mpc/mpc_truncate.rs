use crate::custom_ops::CustomOperationBody;
use crate::data_types::{array_type, scalar_size_in_bits, scalar_type, ScalarType, Type, BIT};
use crate::data_values::Value;
use crate::errors::Result;
use crate::graphs::{Context, Graph, Node, NodeAnnotation};
use crate::mpc::mpc_compiler::{check_private_tuple, KEY_LENGTH, PARTIES};

use serde::{Deserialize, Serialize};

fn get_unsigned_counterpart(st: ScalarType) -> ScalarType {
    if !st.get_signed() {
        return st;
    }

    ScalarType {
        signed: false,
        modulus: st.get_modulus(),
    }
}

/// Truncate MPC operation for public and private data.
///
/// In contrast to plaintext Truncate, this operation might introduce 2 types of errors:
/// 1. 1 bit of additive error in LSB.
///    This bit comes from the fact that truncating the addends of the sum a = b + c by d bits
///    can remove a carry bit propagated to the (d+1)-th bit of the sum.
///    E.g., truncating the addends of 2 = 1 + 1 by 2 results in 1/2 + 1/2 = 0 != 2/2.
/// 2. Additive error in MSBs.
///    Since addition is done modulo 2^m, every sum can be written as a = b + c +- k * 2^m with k in {0,1}.
///    But the truncation result is b/scale + c/scale = (a + k * 2^m)/scale. If k = 1, the error is 2^m/scale.
///    The probability of this error is
///    * 1 - (a + 1) / 2^m for unsigned types,
///    * (|a| - 1) / m, if a < 0 and (a + 1) / m, if a >= 0 for signed types.  
///    Therefore, this operation supports only signed types with a warning
///    that it fails with probability < 2^(l-m) when |a| < 2^l.
///
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub(super) struct TruncateMPC {
    pub scale: u64,
}

#[typetag::serde]
impl CustomOperationBody for TruncateMPC {
    fn instantiate(&self, context: Context, argument_types: Vec<Type>) -> Result<Graph> {
        if argument_types.len() == 1 {
            if let Type::Array(_, st) | Type::Scalar(st) = argument_types[0].clone() {
                if !st.get_signed() {
                    return Err(runtime_error!(
                        "Only signed types are supported by TruncateMPC"
                    ));
                }
                let g = context.create_graph()?;
                let input = g.input(argument_types[0].clone())?;
                let o = if self.scale == 1 {
                    // Do nothing if scale is 1
                    input
                } else {
                    input.truncate(self.scale)?
                };
                o.set_as_output()?;
                g.finalize()?;
                return Ok(g);
            } else {
                // Panics since:
                // - the user has no direct access to this function.
                // - the MPC compiler should pass the correct number of arguments
                // and this panic should never happen.
                panic!("Inconsistency with type checker");
            }
        }
        if argument_types.len() != 2 {
            // Panics since:
            // - the user has no direct access to this function.
            // - the MPC compiler should pass the correct number of arguments
            // and this panic should never happen.
            panic!("TruncateMPC should have either 1 or 2 inputs.");
        }

        if let (Type::Tuple(v0), Type::Tuple(v1)) =
            (argument_types[0].clone(), argument_types[1].clone())
        {
            check_private_tuple(v0)?;
            check_private_tuple(v1)?;
        } else {
            // Panics since:
            // - the user has no direct access to this function.
            // - the MPC compiler should pass the correct number of arguments
            // and this panic should never happen.
            panic!("TruncateMPC should have a private tuple and a tuple of keys as input");
        }

        let t = argument_types[0].clone();
        let input_t = if let Type::Tuple(t_vec) = t.clone() {
            (*t_vec[0]).clone()
        } else {
            panic!("Shouldn't be here");
        };
        if !input_t.get_scalar_type().get_signed() {
            return Err(runtime_error!(
                "Only signed types are supported by TruncateMPC"
            ));
        }

        let g = context.create_graph()?;
        let input_node = g.input(t)?;

        let prf_type = argument_types[1].clone();
        let prf_keys = g.input(prf_type)?;

        // Do nothing if scale is 1.
        if self.scale == 1 {
            input_node.set_as_output()?;
            g.finalize()?;
            return Ok(g);
        }

        // Generate shares of a random value r = PRF_k(v) where k is known to parties 1 and 2 (it's the last key in the key triple).
        let prf_key_parties_12 = prf_keys.tuple_get(PARTIES as u64 - 1)?;
        let random_node = g.prf(prf_key_parties_12, 0, input_t)?;

        let mut result_shares = vec![];
        // 1st share of the result is the truncated 1st share of the input
        let res0 = input_node.tuple_get(0)?.truncate(self.scale)?;
        result_shares.push(res0);
        // 2nd share of the results is the truncated sum of the 2nd and 3rd input shares minus r
        let res1 = input_node
            .tuple_get(1)?
            .add(input_node.tuple_get(2)?)?
            .truncate(self.scale)?
            .subtract(random_node.clone())?;
        let res1_sent = res1.nop()?;
        // 2nd share should be sent to party 0
        res1_sent.add_annotation(NodeAnnotation::Send(1, 0))?;
        result_shares.push(res1_sent);
        // 3rd share of the result is the random value r
        result_shares.push(random_node);

        g.create_tuple(result_shares)?.set_as_output()?;

        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        format!("TruncateMPC({})", self.scale)
    }
}

/// Truncate MPC operation for public and private data by a power of 2.
///
/// Signed input integers must be from the range [-modulus/4, modulus/4)
/// and unsigned integers must be in the range [0, modulus/2) where modulus is the modulus of the input scalar type.
///  
/// This algorithm returns floor(x/2^k) + w where w = 1 with probability (x mod 2^k)/2^k, otherwise w=0.
/// So the result is biased to round(x/2^k).
///
/// The corresponding protocol is described [here](https://eprint.iacr.org/2019/131.pdf#page=10) and runs as follows.
///     0. The below protocol works correctly for integers in the range [0, modulus/2).
///        For signed inputs, we add modulus/4 to input resulting in [0, modulus/2).
///        For correctness, we should remove modulus/2^(k+2) after truncation since
///        Truncate(input + modulus/4, 2^k) = Truncate(input, 2^k) + modulus/2^(k+2).
///
/// Let x = (x0, x1, x2) is the 2-out-of-3 sharing of the (possibly, shifted) input.
/// k_2 is a PRF key that is held only by party 2.
/// k_02 is a PRF key that is held only by parties 0 and 2.
/// k_12 is a PRF key that is held only by parties 1 and 2.
/// The keys k_02 and k_12 are re-used multiplication keys.
///     1. Party 2 generates a random integer r of the input scalar type.
///     2. Party 2 extracts the MSB of r in the arithmetic form (r_msb).
///     3. Party 2 removes the MSB of r and truncates the result by k bits (r_truncated = sum_(i=k)^(s-2) r_i * 2^(i-k) where s is the bitsize of the input scalar type)
///     4. Party 2 creates 2-out-of-2 shares of r, r_msb and r_truncated.
///        Such shares for a value val have the form (val0, val1) such that val = val0 + val1.
///        The corresponding share val0 = PRF(k_02, iv_val0) of the aforementioned 3 values is generated by parties 0 and 2.
///        The second share val1 = val - val0 is computed by party 2 and then it is sent to party 1.
///     5. Parties 0 and 2 compute y0 = PRF(key_02, iv_y0).
///        Parties 1 and 2 compute y2 = PRF(key_12, iv_y2).
///        The pair (y0, y2) is a 2-out-of-3 share of the output known to party 2.
///     6. Parties 0 and 1 create a 2-out-of-2 share of the input x.
///        To obtain its share, party 0 sums its 2-out-of-3 shares to get z0 = x0 + x1.
///        Party 1 takes z1 = x2.
///     7. Given r from party 2, parties 0 and 1 compute 2-out-of-2 shares of c = x + r via c0 = z0 + r0 and c1 = z1 + r1.
///     8. Parties 0 and 1 reveal c to each other and compute c_truncated_mod = (c/2^k) mod 2^(s-k-1).
///        This is c truncated by k bits without its MSB.
///     9. Parties 0 and 1 compute the MSB of c via c/2^(s-1).
///     10. Parties 0 and 1 compute 2-out-of-2 shares of b = r_msb XOR c_msb using the following expressions:
///             b0 = r_msb0 + c_msb - 2 * c_msb * r_msb0,
///             b1 = r_msb1 - 2 * c_msb * r_msb1.
///         Note that b0 + b1 = r_msb + c_msb - 2*c_msb*r_msb = r_msb XOR c_msb.
///         All the above operations can be done locally as c_msb is known to parties 0 and 1.
///     11. Parties 0 and 1 compute 2-out-of-2 shares of y' = c_truncated_mod - r_truncated + b * 2^(st_size-1-k).
///         This value is equal to the desired result floor(x/2^k) + w.
///     12. Party 0 masks y'0 with a random value y0 from party 2 as y_tilde0 = y'0 - y0 and sends it to party 1.
///     13. Party 1 masks y'1 with a random value y2 from party 2 as y_tilde1 = y'1 - y2 and sends it to party 0.
///     14. Parties 0 and 1 compute y1 = y_tilde0 + y_tilde1 = y' - y0 - y2.
///         Together with y0 and y2 this value constitute the sharing of the truncation output.
///     14!. If input is signed, we should remove modulus/2^(k+2) after truncation since
///          Truncate(input + modulus/4, 2^k) = Truncate(input, 2^k) + modulus/2^(k+2) as in Step 0.
///     15. The protocol returns (y0, y1, y2).
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub(super) struct TruncateMPC2K {
    pub k: u64,
}

#[typetag::serde]
impl CustomOperationBody for TruncateMPC2K {
    fn instantiate(&self, context: Context, argument_types: Vec<Type>) -> Result<Graph> {
        if argument_types.len() == 1 {
            if let Type::Array(_, _) | Type::Scalar(_) = argument_types[0].clone() {
                let g = context.create_graph()?;
                let input = g.input(argument_types[0].clone())?;
                let o = if self.k == 0 {
                    // Do nothing if scale is 1
                    input
                } else {
                    input.truncate(1 << self.k)?
                };
                o.set_as_output()?;
                g.finalize()?;
                return Ok(g);
            } else {
                // Panics since:
                // - the user has no direct access to this function.
                // - the MPC compiler should pass the correct number of arguments
                // and this panic should never happen.
                panic!("Inconsistency with type checker");
            }
        }
        if argument_types.len() != 3 {
            // Panics since:
            // - the user has no direct access to this function.
            // - the MPC compiler should pass the correct number of arguments
            // and this panic should never happen.
            panic!("TruncateMPC2K should have 3 inputs.");
        }
        if let Type::Tuple(v0) = argument_types[0].clone() {
            check_private_tuple(v0)?;
        } else {
            if !argument_types[0].is_array() && !argument_types[0].is_scalar() {
                // Panics since:
                // - the user has no direct access to this function.
                // - the MPC compiler should pass the correct number of arguments
                // and this panic should never happen.
                panic!("Inconsistency with type checker");
            }
            let g = context.create_graph()?;
            let input = g.input(argument_types[0].clone())?;
            let o = input.truncate(1 << self.k)?;
            o.set_as_output()?;
            g.finalize()?;
            return Ok(g);
        }

        // Check PRF keys
        let key_type = array_type(vec![KEY_LENGTH], BIT);
        if let Type::Tuple(v0) = argument_types[1].clone() {
            check_private_tuple(v0.clone())?;
            for t in v0 {
                if *t != key_type {
                    // Panics since:
                    // - the user has no direct access to this function.
                    // - the MPC compiler should pass the correct number of arguments
                    // and this panic should never happen.
                    panic!("PRF key is of a wrong type");
                }
            }
        } else {
            // Panics since:
            // - the user has no direct access to this function.
            // - the MPC compiler should pass the correct number of arguments
            // and this panic should never happen.
            panic!("PRF key is of a wrong type");
        }
        if argument_types[2] != key_type {
            // Panics since:
            // - the user has no direct access to this function.
            // - the MPC compiler should pass the correct number of arguments
            // and this panic should never happen.
            panic!("PRF key is of a wrong type");
        }

        let t = argument_types[0].clone();
        let input_t = if let Type::Tuple(t_vec) = t.clone() {
            (*t_vec[0]).clone()
        } else {
            panic!("Shouldn't be here");
        };
        if !input_t.is_array() && !input_t.is_scalar() {
            // Panics since:
            // - the user has no direct access to this function.
            // - the MPC compiler should pass the correct number of arguments
            // and this panic should never happen.
            panic!("Inconsistency with type checker");
        }

        let g = context.create_graph()?;
        let input_node = g.input(t)?;

        // PRF keys
        let prf_mul_type = argument_types[1].clone();
        let prf_mul_keys = g.input(prf_mul_type)?;
        let prf_truncate_type = argument_types[2].clone();
        // PRF key k_2
        let key_2 = g.input(prf_truncate_type)?;

        if self.k == 0 {
            input_node.set_as_output()?;
            g.finalize()?;
            return Ok(g);
        }
        // PRF key k_02, this is the last key in the multiplication PRF key triple
        let key_02 = prf_mul_keys.tuple_get(0)?;
        // PRF key k_12, this is the second key in the multiplication PRF key triple
        let key_12 = prf_mul_keys.tuple_get(2)?;

        let st = input_t.get_scalar_type();
        let st_size = scalar_size_in_bits(st.clone());

        let x0 = {
            let share = input_node.tuple_get(0)?;
            // 0. The below protocol works correctly for integers in the range [0, modulus/2).
            //    For signed inputs, we add modulus/4 to input resulting in input + modulus/4 in [0, modulus/2)
            //    For correctness, we should remove modulus/2^(k+2) after truncation since
            //    Truncate(input + modulus/4, 2^k) = Truncate(input, 2^k) + modulus/2^(k+2)
            if st.get_signed() {
                // modulus/4
                let mod_fraction = g.constant(
                    scalar_type(st.clone()),
                    Value::from_scalar(1u64 << (st_size - 2), st.clone())?,
                )?;
                share.add(mod_fraction)?
            } else {
                share
            }
        };
        let x1 = input_node.tuple_get(1)?;
        let x2 = input_node.tuple_get(2)?;

        // 1. Party 2 generates a random integer r of the input scalar type.
        let r = g.prf(key_2, 0, input_t.clone())?;

        let unsigned_st = get_unsigned_counterpart(st.clone());
        // 2. Party 2 extracts the MSB of r in the arithmetic form (r_msb).
        let r_msb = {
            // (0,0, ..., 1)
            let mask = g
                .constant(
                    scalar_type(unsigned_st.clone()),
                    Value::from_scalar(1u64 << (st_size - 1), unsigned_st.clone())?,
                )?
                .a2b()?;
            // (0,0, ..., r_(st_size-1)) -> r_(st_size-1)*2^(st_size-1) as unsigned integer
            let r_msb_scaled = r.a2b()?.multiply(mask)?.b2a(unsigned_st.clone())?;
            // (r_(st_size-1), 0, ..., 0) -> r_(st_size-1) of st type
            r_msb_scaled
                .truncate(1 << (st_size - 1))?
                .a2b()?
                .b2a(st.clone())?
        };

        // 3. Party 2 removes the MSB of r and truncates the result by k bits (r_truncated = sum_(i=k)^(st_size-2) r_i * 2^(i-k))
        let r_truncated = {
            // (0, ..., 0, 1, ..., 1, 0, ..., 0) to extract r_k, r_(k+1), ..., r_(st_size-2)
            let mask = g
                .constant(
                    scalar_type(unsigned_st.clone()),
                    Value::from_scalar(
                        (1u64 << (st_size - 1)) - (1u64 << self.k),
                        unsigned_st.clone(),
                    )?,
                )?
                .a2b()?;
            // r_k + r_(k+1) * 2 + ... + r_(st_size-2) * 2^(st_size-2-k)
            r.a2b()?
                .multiply(mask)?
                .b2a(st.clone())?
                .truncate(1 << self.k)?
        };

        // 4. Party 2 creates 2-out-of-2 shares of r, r_msb and r_truncated.
        //    Such shares for a value val have the form (val0, val1) such that val = val0 + val1.
        //    The corresponding share val0 = PRF(k_02, iv) of the aforementioned 3 values is generated by parties 0 and 2.
        //    The second share val1 = val - val0 is computed by party 2 and then it is sent to party 1.
        let share_for_two = |val: Node| -> Result<(Node, Node)> {
            // first share val0 for party 0
            let share0 = g.prf(key_02.clone(), 0, val.get_type()?)?;
            // second share val1 for party 1
            let share1 = val.subtract(share0.clone())?;
            let share1_sent = share1.nop()?;
            share1_sent.add_annotation(NodeAnnotation::Send(2, 1))?;
            Ok((share0, share1_sent))
        };
        let (r0, r1) = share_for_two(r)?;
        let (r_msb0, r_msb1) = share_for_two(r_msb)?;
        let (r_truncated0, r_truncated1) = share_for_two(r_truncated)?;

        // 5. Parties 0 and 2 compute y0 = PRF(key_02, iv).
        //    Parties 1 and 2 compute y2 = PRF(key_12, iv).
        //    The pair (y0, y2) is a 2-out-of-3 share of the output known to party 2.
        let y0 = g.prf(key_02, 0, input_t.clone())?;
        let y2 = g.prf(key_12, 0, input_t)?;

        // 6. Party 0 and Party 1 create a 2-out-of-2 share of the input x.
        //    To obtain its share, party 0 sums its 2-out-of-3 shares to get z0 = x0 + x1. Party 1 takes z1 = x2.
        let z0 = x0.add(x1)?;
        let z1 = x2;

        // 7. Given r from party 2, parties 0 and 1 compute 2-out-of-2 shares of c = x + r via c0 = z0 + r0 and c1 = z1 + r1.
        let c_share0 = z0.add(r0)?;
        let c_share1 = z1.add(r1)?;

        // 8. Parties 0 and 1 reveal c to each other and compute c_truncated_mod = (c/2^k) mod 2^(st_size-k-1).
        //    This is c truncated by k bits without its MSB.
        let c_share0_sent = c_share0.nop()?;
        c_share0_sent.add_annotation(NodeAnnotation::Send(0, 1))?;
        let c_share1_sent = c_share1.nop()?;
        c_share1_sent.add_annotation(NodeAnnotation::Send(1, 0))?;
        let c = c_share0_sent.add(c_share1_sent)?;
        // Interpret c as unsigned integer and truncate
        // (c / scale) mod 2^(st_size-1-k)
        let c_truncated = c
            .a2b()?
            .b2a(unsigned_st.clone())?
            .truncate(1 << self.k)?
            .a2b()?
            .b2a(st.clone())?;
        let c_truncated_mod = {
            // (1,1, ..., 1, 0, ..., 0) to perform mod 2^(st_size-1-k)
            let mask = g
                .constant(
                    scalar_type(st.clone()),
                    Value::from_scalar((1u64 << (st_size - 1 - self.k)) - 1, st.clone())?,
                )?
                .a2b()?;
            c_truncated.a2b()?.multiply(mask)?.b2a(st.clone())?
        };

        // 9. Parties 0 and 1 compute the MSB of c via c/2^(st_size-1).
        let c_msb = c
            .a2b()?
            .b2a(unsigned_st)?
            .truncate(1 << (st_size - 1))?
            .a2b()?
            .b2a(st.clone())?;

        // 10. Parties 0 and 1 compute 2-out-of-2 shares of b = r_msb XOR c_msb using the following expressions:
        //             b0 = r_msb0 + c_msb - 2 * c_msb * r_msb0,
        //             b1 = r_msb1 - 2 * c_msb * r_msb1.
        //     Note that b0 + b1 = r_msb + c_msb - 2*c_msb*r_msb = r_msb XOR c_msb.
        //     All the above operations can be done locally as c_msb is known to parties 0 and 1.
        let two = g.constant(scalar_type(st.clone()), Value::from_scalar(2, st.clone())?)?;
        let b0 = r_msb0
            .subtract(r_msb0.multiply(c_msb.clone())?.multiply(two.clone())?)?
            .add(c_msb.clone())?;
        let b1 = r_msb1.subtract(r_msb1.multiply(c_msb)?.multiply(two)?)?;

        // 11. Parties 0 and 1 compute 2-out-of-2 shares of y' = c_truncated_mod - r_truncated + b * 2^(st_size-1-k).
        //     This value is equal to the desired result floor(x/2^k) + w.
        // 2^(st_size-1-k)
        let power2 = g.constant(
            scalar_type(st.clone()),
            Value::from_scalar(1u64 << (st_size - 1 - self.k), st.clone())?,
        )?;
        // y' = c_truncated_mod - r_truncated + b * 2^(st_size-1-k)
        // This is 2-out-of-2 sharing of the result
        let y_prime0 = b0
            .multiply(power2.clone())?
            .subtract(r_truncated0)?
            .add(c_truncated_mod)?;
        let y_prime1 = b1.multiply(power2)?.subtract(r_truncated1)?;

        // 12. Party 0 masks y'0 with a random value y0 from party 2 as y_tilde0 = y'0 - y0 and sends it to party 1.
        let y_tilde0 = y_prime0.subtract(y0.clone())?;
        let y_tilde0_sent = y_tilde0.nop()?;
        y_tilde0_sent.add_annotation(NodeAnnotation::Send(0, 1))?;
        // 13. Party 1 masks y'1 with a random value y2 from party 2 as y_tilde1 = y'1 - y2 and sends it to party 0.
        let y_tilde1 = y_prime1.subtract(y2.clone())?;
        let y_tilde1_sent = y_tilde1.nop()?;
        y_tilde1_sent.add_annotation(NodeAnnotation::Send(1, 0))?;

        // 14. Parties 0 and 1 compute y1 = y_tilde0 + y_tilde1 = y' - y0 - y2.
        //     Together with y0 and y2 this value constitute the sharing of the truncation output.
        let y1 = {
            let sum01 = y_tilde0_sent.add(y_tilde1_sent)?;
            if st.get_signed() {
                // 14!. If input is signed, we should remove modulus/2^(k+2) after truncation since
                //      Truncate(input + modulus/4, 2^k) = Truncate(input, 2^k) + modulus/2^(k+2)
                let mod_fraction = g.constant(
                    scalar_type(st.clone()),
                    Value::from_scalar(1u64 << (st_size - 2 - self.k), st)?,
                )?;
                sum01.subtract(mod_fraction)?
            } else {
                sum01
            }
        };

        // 15. The protocol returns (y0, y1, y2).
        g.create_tuple(vec![y0, y1, y2])?.set_as_output()?;

        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        format!("TruncateMPC2K({})", self.k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytes::subtract_vectors_u64;
    use crate::data_types::{array_type, scalar_type, ScalarType, INT64, UINT64};
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::graphs::create_context;
    use crate::inline::inline_ops::{InlineConfig, InlineMode};
    use crate::mpc::mpc_compiler::{prepare_for_mpc_evaluation, IOStatus, PARTIES};

    fn prepare_context(
        t: Type,
        party_id: IOStatus,
        output_parties: Vec<IOStatus>,
        scale: u64,
        inline_config: InlineConfig,
    ) -> Result<Context> {
        let c = create_context()?;
        let g = c.create_graph()?;
        let i = g.input(t)?;
        let o = g.truncate(i, scale)?;
        g.set_output_node(o)?;
        g.finalize()?;
        c.set_main_graph(g)?;
        c.finalize()?;

        prepare_for_mpc_evaluation(c, vec![vec![party_id]], vec![output_parties], inline_config)
    }

    fn prepare_input(input: Vec<u64>, input_status: IOStatus, t: Type) -> Result<Vec<Value>> {
        let mpc_input = match t {
            Type::Scalar(st) => {
                if input_status == IOStatus::Public || matches!(input_status, IOStatus::Party(_)) {
                    return Ok(vec![Value::from_scalar(input[0], st.clone())?]);
                }

                // shares of input = (input - 3, 1, 2)
                let mut shares_vec = vec![];
                shares_vec.push(Value::from_scalar(
                    subtract_vectors_u64(&input, &[3], st.get_modulus())?[0],
                    st.clone(),
                )?);

                for i in 1..PARTIES as u64 {
                    shares_vec.push(Value::from_scalar(i, st.clone())?);
                }
                shares_vec
            }
            Type::Array(_, st) => {
                if input_status == IOStatus::Public || matches!(input_status, IOStatus::Party(_)) {
                    return Ok(vec![Value::from_flattened_array(&input, st.clone())?]);
                }

                // shares of input = (input - 3, 1, 2)
                let mut shares_vec = vec![];
                let threes = vec![3; input.len()];
                let first_share = subtract_vectors_u64(&input, &threes, st.get_modulus())?;
                shares_vec.push(Value::from_flattened_array(&first_share, st.clone())?);

                for i in 1..PARTIES {
                    let share = vec![i; input.len()];
                    shares_vec.push(Value::from_flattened_array(&share, st.clone())?);
                }
                shares_vec
            }
            _ => {
                panic!("Shouldn't be here");
            }
        };

        Ok(vec![Value::from_vector(mpc_input)])
    }

    // output and expected are assumed to be small enough to be converted to i64 slices
    fn compare_truncate_output(
        output: &[u64],
        expected: &[u64],
        equal: bool,
        st: ScalarType,
    ) -> Result<()> {
        if st.get_signed() {
            for (i, out_value) in output.iter().enumerate() {
                let mut dif = (*out_value) as i64 - expected[i] as i64;
                dif = dif.abs();
                if equal && dif > 1 {
                    return Err(runtime_error!("Output is too far from expected"));
                }
                if !equal && dif <= 1 {
                    return Err(runtime_error!("Output is too close to expected"));
                }
            }
        } else {
            for (i, out_value) in output.iter().enumerate() {
                let dif = (*out_value) - expected[i];
                if equal && dif > 1 {
                    return Err(runtime_error!("Output is too far from expected"));
                }
                if !equal && dif <= 1 {
                    return Err(runtime_error!("Output is too close to expected"));
                }
            }
        }

        Ok(())
    }

    fn check_output(
        mpc_graph: Graph,
        inputs: Vec<Value>,
        expected: Vec<u64>,
        output_parties: Vec<IOStatus>,
        t: Type,
    ) -> Result<()> {
        let output = random_evaluate(mpc_graph.clone(), inputs)?;
        let st = t.get_scalar_type();

        if output_parties.is_empty() {
            let out = output.access_vector(|v| {
                let modulus = st.get_modulus();
                let mut res = vec![0; expected.len()];
                for val in v {
                    let arr = match t.clone() {
                        Type::Scalar(_) => {
                            vec![val.to_u64(st.clone())?]
                        }
                        Type::Array(_, _) => val.to_flattened_array_u64(t.clone())?,
                        _ => {
                            panic!("Shouldn't be here");
                        }
                    };
                    for i in 0..expected.len() {
                        res[i] = if let Some(m) = modulus {
                            (res[i] + arr[i]) % m
                        } else {
                            res[i].wrapping_add(arr[i])
                        };
                    }
                }
                Ok(res)
            })?;
            compare_truncate_output(&out, &expected, true, st.clone())?;
        } else {
            assert!(output.check_type(t.clone())?);
            let out = match t.clone() {
                Type::Scalar(_) => vec![output.to_u64(st.clone())?],
                Type::Array(_, _) => output.to_flattened_array_u64(t.clone())?,
                _ => {
                    panic!("Shouldn't be here");
                }
            };
            compare_truncate_output(&out, &expected, true, st.clone())?;
        }

        Ok(())
    }

    fn truncate_helper(st: ScalarType, scale: u64) -> Result<()> {
        let helper = |t: Type,
                      input: Vec<u64>,
                      input_status: IOStatus,
                      output_parties: Vec<IOStatus>,
                      inline_config: InlineConfig|
         -> Result<()> {
            let mpc_context = prepare_context(
                t.clone(),
                input_status.clone(),
                output_parties.clone(),
                scale,
                inline_config,
            )?;
            let mpc_graph = mpc_context.get_main_graph()?;

            let mpc_input = prepare_input(input.clone(), input_status.clone(), t.clone())?;

            let expected = if t.get_scalar_type().get_signed() {
                input
                    .iter()
                    .map(|x| {
                        let val = *x as i64;
                        let res = val / (scale as i64);
                        res as u64
                    })
                    .collect()
            } else {
                input
                    .iter()
                    .map(|x| {
                        let val = *x;
                        let res = val / scale;
                        res
                    })
                    .collect()
            };
            check_output(mpc_graph, mpc_input, expected, output_parties, t.clone())?;

            Ok(())
        };
        let inline_config_simple = InlineConfig {
            default_mode: InlineMode::Simple,
            ..Default::default()
        };
        let helper_runs = |inputs: Vec<u64>, t: Type| -> Result<()> {
            helper(
                t.clone(),
                inputs.clone(),
                IOStatus::Party(2),
                vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
                inline_config_simple.clone(),
            )?;
            helper(
                t.clone(),
                inputs.clone(),
                IOStatus::Shared,
                vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
                inline_config_simple.clone(),
            )?;
            helper(
                t.clone(),
                inputs.clone(),
                IOStatus::Party(2),
                vec![IOStatus::Party(0)],
                inline_config_simple.clone(),
            )?;
            helper(
                t.clone(),
                inputs.clone(),
                IOStatus::Party(2),
                vec![],
                inline_config_simple.clone(),
            )?;
            helper(
                t.clone(),
                inputs.clone(),
                IOStatus::Public,
                vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
                inline_config_simple.clone(),
            )?;
            helper(
                t.clone(),
                inputs.clone(),
                IOStatus::Public,
                vec![],
                inline_config_simple.clone(),
            )?;
            Ok(())
        };
        // This test should fail with a probability depending on input and the number of runs
        let helper_malformed = |inputs: Vec<u64>, t: Type, runs: u64| -> Result<()> {
            for _ in 0..runs {
                helper_runs(inputs.clone(), t.clone())?;
            }
            Ok(())
        };

        helper_runs(vec![0], scalar_type(st.clone()))?;
        helper_runs(vec![1000], scalar_type(st.clone()))?;
        helper_runs(vec![0, 0], array_type(vec![2], st.clone()))?;
        helper_runs(vec![2000, 255], array_type(vec![2], st.clone()))?;
        if scale.is_power_of_two() && !st.get_signed() {
            // 2^63 - 1, this is a maximal UINT64 value that can be truncated without errors by TruncateMPC2K
            helper_runs(vec![(1u64 << 63) - 1], scalar_type(st.clone()))?;
        }

        if st.get_signed() {
            // -1
            helper_runs(vec![u64::MAX], scalar_type(st.clone()))?;
            // -1000
            helper_runs(vec![u64::MAX - 999], scalar_type(st.clone()))?;
            // [-10. -1024]
            helper_runs(
                vec![u64::MAX as u64 - 9, u64::MAX - 1023],
                array_type(vec![2], st.clone()),
            )?;
            if scale.is_power_of_two() {
                // - 2^62, this is a minimal INT32 value that can be truncated without errors by TruncateMPC2K
                helper_runs(vec![1u64 << 62], scalar_type(st.clone()))?;
                // 2^62-1, this is a maximal INT32 value that can be truncated without errors by TruncateMPC2K
                helper_runs(vec![(1u64 << 62) - 1], scalar_type(st.clone()))?;
            }
        }

        // Probabilistic tests of TruncateMPC for big values in absolute size
        if scale != 1 && !scale.is_power_of_two() {
            // 2^63 - 1, should fail with probability 1 - 2^(-40)
            assert!(helper_malformed(vec![i64::MAX as u64], scalar_type(st.clone()), 40).is_err());
            // -2^63, should fail with probability 1 - 2^(-40)
            assert!(helper_malformed(vec![1u64 << 63], scalar_type(st.clone()), 40).is_err());
            // [2^63 - 1, 2^63 - 2]
            assert!(helper_malformed(
                vec![i64::MAX as u64, i64::MAX as u64 - 1],
                array_type(vec![2], st.clone()),
                40
            )
            .is_err());
            // [-2^63, -2^63 + 1]
            assert!(helper_malformed(
                vec![1u64 << 63, (1u64 << 63) + 1],
                array_type(vec![2], st.clone()),
                40
            )
            .is_err());
        }
        Ok(())
    }

    #[test]
    fn test_truncate() {
        truncate_helper(UINT64, 1).unwrap();
        truncate_helper(UINT64, 1 << 3).unwrap();
        truncate_helper(UINT64, 1 << 7).unwrap();
        truncate_helper(UINT64, 1 << 29).unwrap();
        truncate_helper(UINT64, 1 << 31).unwrap();

        truncate_helper(INT64, 1).unwrap();
        truncate_helper(INT64, 15).unwrap();
        truncate_helper(INT64, 1 << 3).unwrap();
        truncate_helper(INT64, 1 << 7).unwrap();
        truncate_helper(INT64, 1 << 29).unwrap();
        truncate_helper(INT64, (1 << 29) - 1).unwrap();

        assert!(truncate_helper(UINT64, 15).is_err());
    }
}
