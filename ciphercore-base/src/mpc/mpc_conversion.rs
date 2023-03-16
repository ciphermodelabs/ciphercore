use crate::custom_ops::{
    run_instantiation_pass, ContextMappings, CustomOperation, CustomOperationBody,
};
use crate::data_types::{array_type, scalar_type, tuple_type, ScalarType, Type, BIT};
use crate::data_values::Value;
use crate::errors::Result;
use crate::graphs::util::simple_context;
use crate::graphs::SliceElement::SubArray;
use crate::graphs::{Context, Graph, Node, NodeAnnotation};
use crate::inline::inline_ops::{
    inline_operations, DepthOptimizationLevel, InlineConfig, InlineMode,
};
use crate::mpc::mpc_arithmetic::{AddMPC, MultiplyMPC};
use crate::mpc::mpc_compiler::{check_private_tuple, compile_to_mpc_graph, PARTIES};
use crate::ops::adder::BinaryAddTransposed;
use crate::ops::utils::{pull_out_bits, pull_out_bits_for_type, put_in_bits, zeros};
use crate::type_inference::a2b_type_inference;

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub(super) struct A2BMPC {}

/// A2B MPC operation for public and private data with the following arguments:
/// 1. data to be converted from the arithmetic representation to the boolean one (public values or private shares);
/// 2. PRF keys for MPC multiplication (only when data is private)
#[typetag::serde]
impl CustomOperationBody for A2BMPC {
    fn instantiate(&self, context: Context, argument_types: Vec<Type>) -> Result<Graph> {
        // If an input is private, i.e. a tuple of 3 elements (a0, a1, a2), then
        // the parties can access the following elements:
        // 1st party -> a0, a1;
        // 2nd party -> a1, a2;
        // 3rd party -> a2, a0.
        if argument_types.len() == 1 {
            if let Type::Array(_, _) | Type::Scalar(_) = argument_types[0].clone() {
                let g = context.create_graph()?;
                let input = g.input(argument_types[0].clone())?;
                g.a2b(input)?.set_as_output()?;
                g.finalize()?;
                return Ok(g);
            } else {
                // Panics since:
                // - the user has no direct access to this function.
                // - the MPC compiler should pass correct arguments
                // and this panic should never happen.
                panic!("Inconsistency with type checker");
            }
        }
        if argument_types.len() != 2 {
            return Err(runtime_error!("A2BMPC should have either 1 or 2 inputs."));
        }

        if let (Type::Tuple(v0), Type::Tuple(v1)) =
            (argument_types[0].clone(), argument_types[1].clone())
        {
            check_private_tuple(v0)?;
            check_private_tuple(v1)?;
        } else {
            return Err(runtime_error!(
                "A2BMPC should have a private tuple and a tuple of keys as input"
            ));
        }

        let t = argument_types[0].clone();
        let input_t = if let Type::Tuple(t_vec) = t.clone() {
            (*t_vec[0]).clone()
        } else {
            return Err(runtime_error!("Shouldn't be here"));
        };

        let bits_t = pull_out_bits_for_type(a2b_type_inference(input_t)?)?;

        // Create helper graphs.
        // They must be generated before the main graph g.
        // Create an MPC graph for the left shift
        let shift_mpc_g = get_left_shift_graph(context.clone(), bits_t.clone())?;
        // Create an MPC graph for the binary adder
        let adder_mpc_g = get_binary_adder_graph(context.clone(), bits_t)?;

        let g = context.create_graph()?;
        let input = g.input(t)?;

        let prf_type = argument_types[1].clone();
        let prf_keys = g.input(prf_type)?;

        // generate shares of arithmetic shares converted to binary strings
        let mut bit_shares = vec![];
        // convert each arithmetic share to a binary array
        let mut input_bits = vec![];
        for i in 0..PARTIES {
            input_bits.push(pull_out_bits(input.tuple_get(i as u64)?.a2b()?)?);
        }

        let zero_bits = g.zeros(input_bits[0].get_type()?)?;
        for share_id in 0..PARTIES {
            let mut bit_share = vec![];
            // for every arithmetic share X create its binary sharing as
            // (X in binary, 0, 0)
            for (party_id, share) in input_bits.iter().enumerate().take(PARTIES) {
                let bit_share_arith = if share_id == party_id {
                    share.clone()
                } else {
                    zero_bits.clone()
                };
                bit_share.push(bit_share_arith);
            }
            let bit_share_tuple = g.create_tuple(bit_share)?;
            bit_shares.push(bit_share_tuple);
        }
        // Sum binary shares
        let transposed_output = add_3_bitstrings(
            g.clone(),
            adder_mpc_g,
            shift_mpc_g,
            bit_shares[0].clone(),
            bit_shares[1].clone(),
            bit_shares[2].clone(),
            prf_keys,
        )?;
        let mut output = vec![];
        for i in 0..PARTIES {
            output.push(put_in_bits(transposed_output.tuple_get(i as u64)?)?);
        }
        let o = g.create_tuple(output)?;
        o.set_as_output()?;
        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        "A2BMPC".to_owned()
    }
}

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub(super) struct B2AMPC {
    pub st: ScalarType,
}

/// B2A MPC operation for public and private data with the following arguments:
/// 1. data to be converted from the boolean representation to the arithmetic one (public values or private shares);
/// 2. PRF keys for MPC multiplication (only when data is private)
/// 3. special PRF keys for B2A (only when data is private)
/// Special PRF keys (2 triples of keys) have the following access pattern:
/// special PRF keys = ((k_11, k_12, k_13), (k_21, k_22, k_23)) where
/// k_11, k_12, k_13, k_21, k_22 are known to party 0,
/// k_12, k_13, k_21, k_22, k_23 are known to party 1,
/// all these keys are known to party 2.
/// TODO: make sure that such access pattern is preserved in computation and networking
#[typetag::serde]
impl CustomOperationBody for B2AMPC {
    fn instantiate(&self, context: Context, argument_types: Vec<Type>) -> Result<Graph> {
        // If an input is private, i.e. a tuple of 3 elements (a0, a1, a2), then
        // the parties can access the following elements:
        // 1st party -> a0, a1;
        // 2nd party -> a1, a2;
        // 3rd party -> a2, a0.
        if argument_types.len() == 1 {
            if let Type::Array(_, _) | Type::Scalar(_) = argument_types[0].clone() {
                let g = context.create_graph()?;
                let input = g.input(argument_types[0].clone())?;
                g.b2a(input, self.st.clone())?.set_as_output()?;
                g.finalize()?;
                return Ok(g);
            } else {
                // Panics since:
                // - the user has no direct access to this function.
                // - the MPC compiler should pass correct arguments
                // and this panic should never happen.
                panic!("Inconsistency with type checker");
            }
        }
        if argument_types.len() != 3 {
            return Err(runtime_error!("B2AMPC should have either 1 or 3 inputs."));
        }

        if let (Type::Tuple(v0), Type::Tuple(v1), Type::Tuple(v2)) = (
            argument_types[0].clone(),
            argument_types[1].clone(),
            argument_types[2].clone(),
        ) {
            check_private_tuple(v0)?;
            check_private_tuple(v1)?;
            // there should be an additional pair of PRF key triples
            if v2.len() != 2 {
                return Err(runtime_error!(
                    "There should be {} PRF key triples, but {} provided",
                    2,
                    v2.len()
                ));
            }
            if *v2[0] != *v2[1] {
                return Err(runtime_error!("PRF keys should be of the same type"));
            }
            if let Type::Tuple(sub_v) = (*v2[0]).clone() {
                check_private_tuple(sub_v)?;
            } else {
                return Err(runtime_error!(
                    "Special PRF keys for B2A should be a tuple of tuples"
                ));
            }
        } else {
            return Err(runtime_error!(
                "B2AMPC should have a private tuple and a tuple of keys as input"
            ));
        }

        let t = argument_types[0].clone();
        let input_t = if let Type::Tuple(t_vec) = t.clone() {
            pull_out_bits_for_type((*t_vec[0]).clone())?
        } else {
            panic!("Shouldn't be here");
        };

        // Create helper graphs.
        // They must be generated before the main graph g.
        // Create an MPC graph for the left shift
        let shift_mpc_g = get_left_shift_graph(context.clone(), input_t.clone())?;
        // Create an MPC graph for the binary adder
        let adder_mpc_g = get_binary_adder_graph(context.clone(), input_t.clone())?;

        let g = context.create_graph()?;
        let input = g.input(t)?;
        let mut transposed_input = vec![];
        for i in 0..PARTIES {
            transposed_input.push(pull_out_bits(input.tuple_get(i as u64)?)?);
        }
        let input = g.create_tuple(transposed_input)?;

        let prf_for_mul_type = argument_types[1].clone();
        let prf_for_mul_keys = g.input(prf_for_mul_type)?;

        let prf_for_random_type = argument_types[2].clone();
        let prf_for_random_keys = g.input(prf_for_random_type)?;

        // Generate binary shares of values equal to output arithmetic shares
        let mut bit_shares = vec![];
        let mut random_shares = vec![];

        // In the 1st iteration party 0 and party 2 generate a random binary share -x_0.
        // In the 2nd iteration party 1 and party 2 generate a random binary share -x_2.
        for share_id in 0..(PARTIES - 1) as u64 {
            let mut random_share = vec![];
            // Extract one triple of PRF keys
            let prf_key_triple = prf_for_random_keys.tuple_get(share_id)?;
            // Create shares of a random value
            for i in 0..PARTIES as u64 {
                let prf_key = prf_key_triple.tuple_get(i)?;
                let random_value = g.prf(prf_key, 0, input_t.clone())?;
                random_share.push(random_value);
            }
            random_shares.push(g.create_tuple(random_share.clone())?);
            // XOR shares of a random value
            let bit_share = random_share[0]
                .add(random_share[1].clone())?
                .add(random_share[2].clone())?;
            bit_shares.push(bit_share);
        }

        // Now we have negations of two out of 3 shares of the conversion result in the binary form: -x_0 and -x_2.
        // To find the missing share of the final result, x_1, compute x_1 = Add(x, -x_0, -x_2).
        let last_share_shared = add_3_bitstrings(
            g.clone(),
            adder_mpc_g,
            shift_mpc_g,
            input,
            random_shares[0].clone(),
            random_shares[1].clone(),
            prf_for_mul_keys,
        )?;

        // Reveal the last share x_1 to parties 0 and 1.
        // Party 0 needs the 2nd share.
        // Party 1 needs the 0th share.

        // Party 0 sends the 0th share of x_1 to party 1
        let x1_share0 = last_share_shared.tuple_get(0)?.nop()?;
        x1_share0.add_annotation(NodeAnnotation::Send(0, 1))?;
        // Party 1 sends the 2nd share of x_1 to party 0
        let x1_share2 = last_share_shared.tuple_get(2)?.nop()?;
        x1_share2.add_annotation(NodeAnnotation::Send(1, 0))?;

        // Sum all the shares of x_1
        let mut x1_revealed = last_share_shared.tuple_get(1)?;
        x1_revealed = x1_revealed.add(x1_share0)?.add(x1_share2)?;

        // [-x_0, x_1, -x_2]
        bit_shares.insert(1, x1_revealed);

        let zero = g.constant(
            scalar_type(self.st.clone()),
            Value::zero_of_type(scalar_type(self.st.clone())),
        )?;

        // Convert -x_0, x_1 and -x_2 from binary to arithmetic
        let mut arith_shares = vec![];
        for share in bit_shares.into_iter() {
            arith_shares.push(put_in_bits(share)?.b2a(self.st.clone())?);
        }
        // Negate -x_0 and -x_2 to x_0 and x_2, respectively
        arith_shares[0] = zero.subtract(arith_shares[0].clone())?;
        arith_shares[2] = zero.subtract(arith_shares[2].clone())?;

        let o = g.create_tuple(arith_shares)?;

        o.set_as_output()?;
        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        format!("B2AMPC({})", self.st)
    }
}

fn get_left_shift_graph(context: Context, bits_t: Type) -> Result<Graph> {
    let shift_g = context.create_graph()?;
    {
        let tuple_bits_t = tuple_type(vec![bits_t.clone(); PARTIES]);
        let input = shift_g.input(tuple_bits_t)?;
        let shape = bits_t.get_shape();
        let mut new_shape = shape;
        new_shape[0] = 1;
        let t = array_type(new_shape, BIT);
        let zero = zeros(&shift_g, t)?;

        let mut result_shares = vec![];
        for i in 0..PARTIES {
            let share = input.tuple_get(i as u64)?;

            let rows = shift_g.concatenate(
                vec![
                    zero.clone(),
                    share.get_slice(vec![SubArray(None, Some(-1), None)])?,
                ],
                0,
            )?;
            result_shares.push(rows);
        }
        let o = shift_g.create_tuple(result_shares)?;

        o.set_as_output()?;
        shift_g.finalize()?;
    }

    Ok(shift_g)
}

fn get_binary_adder_graph(context: Context, bits_t: Type) -> Result<Graph> {
    // Binary adder
    let adder_context = simple_context(|g| {
        let input1 = g.input(bits_t.clone())?;
        let input2 = g.input(bits_t)?;
        g.custom_op(
            CustomOperation::new(BinaryAddTransposed {
                overflow_bit: false,
            }),
            vec![input1, input2],
        )
    })?;
    let instantiated_adder_context = run_instantiation_pass(adder_context)?.get_context();
    let inlined_adder_context = inline_operations(
        instantiated_adder_context,
        InlineConfig {
            default_mode: InlineMode::DepthOptimized(DepthOptimizationLevel::Default),
            ..Default::default()
        },
    )?;

    let mut context_map = ContextMappings::default();

    // Compile adder to MPC
    let adder_g_inlined = inlined_adder_context.get_main_graph()?;
    let adder_mpc_g =
        compile_to_mpc_graph(adder_g_inlined, vec![true, true], context, &mut context_map)?;
    Ok(adder_mpc_g)
}

/// Adds 3 bit strings contained in nodes a, b and c of a graph g.
/// This function calls 2 graphs:
/// - adder_g that is the full adder graph for 2 bit strings,
/// - shift_g that performs the left shift of a bit string by 1 position.
fn add_3_bitstrings(
    g: Graph,
    adder_g: Graph,
    shift_g: Graph,
    a: Node,
    b: Node,
    c: Node,
    prf_for_mul_keys: Node,
) -> Result<Node> {
    // Reduce addition of three values to two values via the full adder circuit.
    // Given 3 bitstrings a, b and c, Add(a,b,c) = Add(carry << 1, xor_123),
    // where carry = a AND b XOR c AND (a XOR b), xor_123 = a XOR b XOR c
    // and Add is the binary adder.
    // a XOR b
    let xor_12 = g.custom_op(CustomOperation::new(AddMPC {}), vec![a.clone(), b.clone()])?;
    // a AND b
    let and_12 = g.custom_op(
        CustomOperation::new(MultiplyMPC {}),
        vec![a, b, prf_for_mul_keys.clone()],
    )?;
    // (a XOR b) AND c
    let xor_12_and_3 = g.custom_op(
        CustomOperation::new(MultiplyMPC {}),
        vec![xor_12.clone(), c.clone(), prf_for_mul_keys.clone()],
    )?;
    // a XOR b XOR c
    let xor_123 = g.custom_op(CustomOperation::new(AddMPC {}), vec![xor_12, c])?;
    // (a AND b) XOR (a XOR b) AND c
    let carry = g.custom_op(CustomOperation::new(AddMPC {}), vec![and_12, xor_12_and_3])?;

    // carry << 1
    let shifted_carry = g.call(shift_g, vec![carry])?;

    // xor_123 + (carry << 1) = Add(a,b,c)
    g.call(adder_g, vec![prf_for_mul_keys, xor_123, shifted_carry])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytes::subtract_vectors_u64;
    use crate::data_types::{array_type, ScalarType, INT32, UINT32};
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::graphs::Operation;
    use crate::inline::inline_ops::{InlineConfig, InlineMode};
    use crate::mpc::mpc_compiler::{prepare_for_mpc_evaluation, IOStatus};
    use crate::type_inference::a2b_type_inference;

    fn prepare_context(
        op: Operation,
        party_id: IOStatus,
        output_parties: Vec<IOStatus>,
        t: Type,
        inline_config: InlineConfig,
    ) -> Result<Context> {
        let input_t = if op == Operation::A2B {
            t.clone()
        } else {
            a2b_type_inference(t.clone())?
        };
        let c = simple_context(|g| {
            let i = g.input(input_t)?;
            g.add_node(vec![i], vec![], op)
        })?;

        prepare_for_mpc_evaluation(c, vec![vec![party_id]], vec![output_parties], inline_config)
    }

    fn prepare_input(
        op: Operation,
        input: Vec<u64>,
        input_status: IOStatus,
        t: Type,
    ) -> Result<Vec<Value>> {
        let mut res = vec![];
        if input_status == IOStatus::Public {
            match t {
                Type::Scalar(st) => {
                    res.push(Value::from_scalar(input[0], st)?);
                }
                Type::Array(_, st) => {
                    res.push(Value::from_flattened_array(&input, st)?);
                }
                _ => {
                    panic!("Shouldn't be here");
                }
            }
            return Ok(res);
        }

        let mut data_input = vec![];
        match t {
            Type::Array(_, st) => {
                if matches!(input_status, IOStatus::Party(_)) {
                    res.push(Value::from_flattened_array(&input, st.clone())?);
                    return Ok(res);
                }
                if let Operation::B2A(_) = op {
                    // shares of input = (input^3, 2, 1)
                    let first_share: Vec<u64> = input.iter().map(|x| (*x) ^ 3).collect();
                    data_input.push(Value::from_flattened_array(&first_share, st.clone())?);
                } else {
                    // shares of input = (input-3, 1, 2)
                    let threes = vec![3; input.len()];
                    let first_share = subtract_vectors_u64(&input, &threes, st.get_modulus())?;
                    data_input.push(Value::from_flattened_array(&first_share, st.clone())?);
                }
                for i in 1..PARTIES {
                    let share = vec![i; input.len()];
                    data_input.push(Value::from_flattened_array(&share, st.clone())?);
                }
            }
            Type::Scalar(st) => {
                if matches!(input_status, IOStatus::Party(_)) {
                    res.push(Value::from_scalar(input[0], st.clone())?);
                    return Ok(res);
                }
                if let Operation::B2A(_) = op {
                    // shares of input = (input^3, 2, 1)
                    let first_share = input[0] ^ 3;
                    data_input.push(Value::from_scalar(first_share, st.clone())?);
                } else {
                    // shares of input = (input-3, 1, 1)
                    let first_share = subtract_vectors_u64(&input, &vec![3], st.get_modulus())?;
                    data_input.push(Value::from_scalar(first_share[0], st.clone())?);
                }
                for i in 1..PARTIES {
                    data_input.push(Value::from_scalar(i, st.clone())?);
                }
            }
            _ => {
                panic!("Shouldn't be here");
            }
        }

        res.push(Value::from_vector(data_input));

        Ok(res)
    }

    fn check_output(
        op: Operation,
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
                        if op == Operation::A2B {
                            res[i] ^= arr[i];
                        } else {
                            res[i] = if let Some(m) = modulus {
                                (res[i] + arr[i]) % m
                            } else {
                                res[i] + arr[i]
                            };
                        }
                    }
                }
                Ok(res)
            })?;
            assert_eq!(out, expected);
        } else {
            assert!(output.check_type(t.clone())?);
            let out = match t.clone() {
                Type::Scalar(_) => vec![output.to_u64(st.clone())?],
                Type::Array(_, _) => output.to_flattened_array_u64(t.clone())?,
                _ => {
                    panic!("Shouldn't be here");
                }
            };
            assert_eq!(out, expected);
        }
        Ok(())
    }

    fn conversion_test(op: Operation, st: ScalarType) -> Result<()> {
        let helper = |input: Vec<u64>,
                      input_status: IOStatus,
                      output_parties: Vec<IOStatus>,
                      inline_config: InlineConfig,
                      t: Type|
         -> Result<()> {
            if let Operation::B2A(st_b2a) = op.clone() {
                if st_b2a != st {
                    panic!("The scalar type of B2A should be equal to the input scalar type");
                }
            }
            let mpc_context = prepare_context(
                op.clone(),
                input_status.clone(),
                output_parties.clone(),
                t.clone(),
                inline_config,
            )?;
            let mpc_graph = mpc_context.get_main_graph()?;

            let inputs = prepare_input(op.clone(), input.clone(), input_status.clone(), t.clone())?;

            check_output(
                op.clone(),
                mpc_graph,
                inputs,
                input.clone(),
                output_parties,
                t.clone(),
            )?;

            Ok(())
        };
        let inline_config_simple = InlineConfig {
            default_mode: InlineMode::Simple,
            ..Default::default()
        };
        let helper_runs = |inputs: Vec<u64>, t: Type| -> Result<()> {
            helper(
                inputs.clone(),
                IOStatus::Party(2),
                vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
                inline_config_simple.clone(),
                t.clone(),
            )?;
            helper(
                inputs.clone(),
                IOStatus::Party(2),
                vec![IOStatus::Party(0), IOStatus::Party(1)],
                inline_config_simple.clone(),
                t.clone(),
            )?;
            helper(
                inputs.clone(),
                IOStatus::Party(2),
                vec![IOStatus::Party(0)],
                inline_config_simple.clone(),
                t.clone(),
            )?;
            helper(
                inputs.clone(),
                IOStatus::Party(2),
                vec![],
                inline_config_simple.clone(),
                t.clone(),
            )?;
            helper(
                inputs.clone(),
                IOStatus::Public,
                vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
                inline_config_simple.clone(),
                t.clone(),
            )?;
            helper(
                inputs.clone(),
                IOStatus::Public,
                vec![],
                inline_config_simple.clone(),
                t.clone(),
            )?;
            Ok(())
        };
        helper_runs(vec![85], scalar_type(st.clone()))?;
        helper_runs(vec![u32::MAX as u64 - 12345677], scalar_type(st.clone()))?;
        helper_runs(vec![2, 85], array_type(vec![2], st.clone()))?;
        helper_runs(vec![0, 255], array_type(vec![2], st.clone()))?;
        helper_runs(
            vec![12345678, u32::MAX as u64 - 12345677],
            array_type(vec![2], st.clone()),
        )?;
        Ok(())
    }

    #[test]
    fn test_a2b_mpc() {
        conversion_test(Operation::A2B, UINT32).unwrap();
        conversion_test(Operation::A2B, INT32).unwrap();
    }

    #[test]
    fn test_b2a_mpc() {
        conversion_test(Operation::B2A(UINT32), UINT32).unwrap();
        conversion_test(Operation::B2A(INT32), INT32).unwrap();
    }
}
