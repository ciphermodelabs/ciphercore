use crate::custom_ops::CustomOperationBody;
use crate::data_types::Type;
use crate::errors::Result;
use crate::graphs::{Context, Graph, NodeAnnotation};
use crate::mpc::mpc_compiler::{check_private_tuple, PARTIES};

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub(super) struct TruncateMPC {
    pub scale: u64,
}

/// Truncate MPC operation for public and private data.
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
        result_shares.push(res1);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytes::subtract_vectors_u64;
    use crate::data_types::{array_type, scalar_type, ScalarType, INT32, UINT32};
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::graphs::create_context;
    use crate::inline::inline_ops::{InlineConfig, InlineMode};
    use crate::mpc::mpc_compiler::{prepare_for_mpc_evaluation, IOStatus};

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
        let m = st.get_modulus().unwrap() as i64;
        for (i, out_value) in output.iter().enumerate() {
            let mut dif = ((*out_value) as i64 - expected[i] as i64) % m;
            dif = if dif >= m / 2 {
                dif - m
            } else if dif < -m / 2 {
                dif + m
            } else {
                dif
            };
            dif = dif.abs();
            if equal && dif > 1 {
                return Err(runtime_error!("Output is too far from expected"));
            }
            if !equal && dif <= 1 {
                return Err(runtime_error!("Output is too close to expected"));
            }
        }
        Ok(())
    }

    fn check_output(
        mpc_graph: Graph,
        inputs: Vec<Value>,
        expected: Vec<u64>,
        output_parties: Vec<IOStatus>,
        is_output_private: bool,
        t: Type,
    ) -> Result<()> {
        let output = random_evaluate(mpc_graph.clone(), inputs)?;
        let st = t.get_scalar_type();
        if is_output_private {
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
                                res[i] + arr[i]
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
        } else {
            let array_output = match t.clone() {
                Type::Scalar(_) => Value::from_scalar(expected[0], st)?,
                Type::Array(_, _) => Value::from_flattened_array(&expected, st)?,
                _ => {
                    panic!("Shouldn't be here");
                }
            };
            assert_eq!(output, array_output);
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

            let is_output_private = input_status != IOStatus::Public;

            let m = t.get_scalar_type().get_modulus().unwrap();
            let expected = input
                .iter()
                .map(|x| {
                    let mut val = *x as i64;
                    if val >= (m / 2) as i64 {
                        val -= m as i64;
                    }
                    let mut res = val / (scale as i64);
                    if res < 0 {
                        res += m as i64;
                    }
                    res as u64
                })
                .collect();
            check_output(
                mpc_graph,
                mpc_input,
                expected,
                output_parties,
                is_output_private,
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
        // -1000
        helper_runs(vec![u32::MAX as u64 - 999], scalar_type(st.clone()))?;

        helper_runs(vec![0, 0], array_type(vec![2], st.clone()))?;
        helper_runs(vec![2000, 255], array_type(vec![2], st.clone()))?;
        // [-10. -1024]
        helper_runs(
            vec![u32::MAX as u64 - 9, u32::MAX as u64 - 1023],
            array_type(vec![2], st.clone()),
        )?;

        // Probabilistic tests for big values in absolute size
        if scale != 1 {
            // 2^31 - 1, should fail with probability 1 - 2^(-40)
            assert!(helper_malformed(vec![i32::MAX as u64], scalar_type(st.clone()), 40).is_err());
            // -2^31, should fail with probability 1 - 2^(-40)
            assert!(helper_malformed(vec![1 << 31], scalar_type(st.clone()), 40).is_err());
            // [2^31 - 1, 2^31 - 2]
            assert!(helper_malformed(
                vec![i32::MAX as u64, i32::MAX as u64 - 1],
                array_type(vec![2], st.clone()),
                40
            )
            .is_err());
            // [-2^31, -2^31 + 1]
            assert!(helper_malformed(
                vec![1 << 31, 1 << 31 + 1],
                array_type(vec![2], st.clone()),
                40
            )
            .is_err());
        }
        Ok(())
    }

    #[test]
    fn test_truncate() {
        truncate_helper(INT32, 1).unwrap();
        truncate_helper(INT32, 8).unwrap();
        truncate_helper(INT32, 80).unwrap();
        truncate_helper(INT32, 1 << 29).unwrap();
        assert!(truncate_helper(UINT32, 80).is_err());
    }
}
