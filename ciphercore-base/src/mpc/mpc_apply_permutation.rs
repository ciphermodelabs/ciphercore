use crate::custom_ops::CustomOperationBody;
use crate::data_types::Type;
use crate::errors::Result;
use crate::graphs::{Context, Graph, Node, NodeAnnotation};
use crate::mpc::mpc_compiler::{check_private_tuple, PARTIES};
use crate::ops::utils::zeros;

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub(super) struct ApplyPermutationMPC {
    pub(crate) inverse_permutation: bool,
    pub(crate) reveal_output: bool,
}

/// ApplyPermutation MPC operation for all public or all private data with the following arguments:
/// 1. single array to be shuffled (public values or private shares);
/// 2. single array -- permutation to apply (public values or private shares);
/// 3. PRF keys for MPC multiplication (only permutation is private).
///
/// Parameters:
/// * inverse_permutation: if true, then inverse permutation is applied.
/// * reveal_output: if true, then output will be revealed to all parties.
#[typetag::serde]
impl CustomOperationBody for ApplyPermutationMPC {
    fn instantiate(&self, context: Context, argument_types: Vec<Type>) -> Result<Graph> {
        if argument_types.len() != 2 && argument_types.len() != 3 {
            return Err(runtime_error!(
                "ApplyPermutationMPC should have either 2 or 3 inputs."
            ));
        }
        let g = context.create_graph()?;
        let input = g.input(argument_types[0].clone())?;
        let perm = g.input(argument_types[1].clone())?;

        if perm.get_type()?.is_array() {
            return self.apply_public_permutation(input, perm, g);
        }

        // Permutation is private.
        if argument_types.len() != 3 {
            return Err(runtime_error!(
                "ApplyPermutationMPC should have prf keys the permutation is private."
            ));
        }

        let prf_keys = g.input(argument_types[2].clone())?;

        if let (Type::Tuple(v1), Type::Tuple(v2)) = (perm.get_type()?, prf_keys.get_type()?) {
            check_private_tuple(v1)?;
            check_private_tuple(v2)?;
        } else {
            return Err(runtime_error!(
                "ApplyPermutationMPC should have a private tuple of permutation and a tuple of keys as input"
            ));
        }

        let shares = if input.get_type()?.is_array() {
            // If an array is public and permutation is private, let's make an array of 'shares'.
            vec![
                input.clone(),
                zeros(&g, input.get_type()?)?,
                zeros(&g, input.get_type()?)?,
            ]
        } else {
            if let Type::Tuple(shape) = input.get_type()? {
                check_private_tuple(shape)?;
            } else {
                return Err(runtime_error!("Unreachable"));
            }
            let mut shares = vec![];
            for i in 0..PARTIES as u64 {
                let share = g.tuple_get(input.clone(), i)?;
                shares.push(share);
            }
            shares
        };

        let n = shares[0].get_type()?.get_shape()[0];
        let prf_keys = vec![
            prf_keys.tuple_get(0)?,
            prf_keys.tuple_get(1)?,
            prf_keys.tuple_get(2)?,
        ];

        // Input:
        // - secret shared input array/ tuple x = x0 + x1 + x2
        // - secret shared random permutation p = p0 * p1 * p2

        // If an input is private, i.e. a tuple of 3 elements (x0, x1, x2), then
        // party 0 -> x0, x1;
        // party 1 -> x1, x2;
        // party 2 -> x2, x0.

        // We are going to apply a permutation, which is the composition of 3 permutations:
        // p = p0 * p1 * p2
        let mut p = vec![];
        for party_id in 0..PARTIES {
            p.push(perm.tuple_get(party_id as u64)?);
        }
        if p[0].get_type()?.get_shape()[0] != n {
            return Err(runtime_error!(
                "ApplyPermutationMPC: input and permutation should have the same first dimension."
            ));
        }
        // To apply inverse permutation, let's notice that if p = p0 * p1 * p2, then p^{-1} = p2^{-1} * p1^{-1} * p0^{-1}.
        // It is easy to show, that to apply p^{-1} to an array x, we can reuse the same protocol with a few modifications:
        // 0. Inverse all permutations.
        // 1. Swap p0 and p2.
        // 2. Swap x0 and x2.
        // 3. Swap parties 0 and 1.
        if self.inverse_permutation {
            for perm in p.iter_mut() {
                *perm = perm.inverse_permutation()?;
            }
        }

        let (p0, p1, p2) = if self.inverse_permutation {
            (p[2].clone(), p[1].clone(), p[0].clone())
        } else {
            (p[0].clone(), p[1].clone(), p[2].clone())
        };
        let (x0, x1, x2) = if self.inverse_permutation {
            (shares[2].clone(), shares[1].clone(), shares[0].clone())
        } else {
            (shares[0].clone(), shares[1].clone(), shares[2].clone())
        };
        let (party_0, party_1, party_2) = if self.inverse_permutation {
            (1, 0, 2)
        } else {
            (0, 1, 2)
        };

        // Protocol needs PRF keys for multiplication, which contains:
        // - k02 = prf_keys[0] known to parties 0 and 2,
        // - k01 = prf_keys[1] known to parties 0 and 1,
        // - k12 = prf_keys[2] known to parties 1 and 2.
        let k02 = if self.inverse_permutation {
            prf_keys[2].clone()
        } else {
            prf_keys[0].clone()
        };
        let k01 = prf_keys[1].clone();
        let t = x0.get_type()?;

        if self.reveal_output {
            // 1. Parties 0 and 2 compute alpha02 = PRF(k02)
            let alpha02 = g.prf(k02, 0, t)?;
            // 2. Party 0 computes b1 = p0(x0 + x1) - alpha02 and sends it to party 1
            let b1 = apply_permutation(x0.add(x1)?, p0.clone())?
                .subtract(alpha02.clone())?
                .nop()?
                .add_annotation(NodeAnnotation::Send(party_0, party_1))?;
            // 3. Party 2 computes b2 = p0(x2) + alpha02 and sends it to party 1.
            let b2 = apply_permutation(x2, p0)?
                .add(alpha02)?
                .nop()?
                .add_annotation(NodeAnnotation::Send(party_2, party_1))?;
            // 4. Party 1 computes c = p2(p1(b1 + b2)) and sends it to party 0 and 2.
            let c = apply_permutation(apply_permutation(b1.add(b2)?, p1)?, p2)?
                .nop()?
                .add_annotation(NodeAnnotation::Send(party_1, party_0))?
                .nop()?
                .add_annotation(NodeAnnotation::Send(party_1, party_2))?;
            c.set_as_output()?;

            // Check:
            // c = b1 + b2
            // = p2(p1(p0(a0 + a1) - alpha02 + p0(a2) + alpha02))
            // = p2(p1(p0(a)))
            // =p(a)
        } else {
            // 1. Parties 0 and 2 compute alpha02 = PRF(k02)
            let alpha02 = g.prf(k02.clone(), 0, t.clone())?;
            // 2. Party 2 computes b0 = p0(x2) + alpha02 and sends it to party 1.
            let b0 = apply_permutation(x2, p0.clone())?
                .add(alpha02.clone())?
                .nop()?
                .add_annotation(NodeAnnotation::Send(party_2, party_1))?;
            // 3. Party 1 computes c1 = p2(p1(b0))
            let c1 = apply_permutation(apply_permutation(b0, p1.clone())?, p2.clone())?;
            // 4. Party 0 computes c0 = p1(p0(x0 + x1)) - p1(alpha02)
            let c0 = apply_permutation(apply_permutation(x0.add(x1)?, p0)?, p1.clone())?
                .subtract(apply_permutation(alpha02, p1)?)?;
            // 5. Parties 0 and 1 compute alpha01 = PRF(k01)
            let alpha01 = g.prf(k01.clone(), 0, t.clone())?;
            // 6. Party 0 computes c2 = c0 - alpha01 and sends it to party 2.
            let c2 = c0
                .subtract(alpha01.clone())?
                .nop()?
                .add_annotation(NodeAnnotation::Send(party_0, party_2))?;
            // 7. Party 1 computes d1 = c1 + p2(alpha01)
            let d1 = c1.add(apply_permutation(alpha01, p2.clone())?)?;
            // 8. Party 2 computes d2 = p2(c2)
            let d2 = apply_permutation(c2, p2)?;
            // 9. Parties 0 and 1 compute beta01 = PRF(k01)
            let beta01 = g.prf(k01, 0, t.clone())?;
            // 10. Parties 0 and 2 compute beta02 = PRF(k02)
            let beta02 = g.prf(k02, 0, t)?;
            // 11. Party 1 computes t1 = d1 - beta01 and sends it to party 2
            let t1 = d1
                .subtract(beta01.clone())?
                .nop()?
                .add_annotation(NodeAnnotation::Send(party_1, party_2))?;
            // 12. Party 2 computes t2 = d2 - beta02 and sends it to party 1
            let t2 = d2
                .subtract(beta02.clone())?
                .nop()?
                .add_annotation(NodeAnnotation::Send(party_2, party_1))?;
            // 13. Parties 1 and 2 compute beta12 = t1 + t2
            let beta12 = t1.add(t2)?;
            // Parties output (beta02, beta01, beta12)
            let shuffled = if self.inverse_permutation {
                vec![beta12, beta01, beta02]
            } else {
                vec![beta02, beta01, beta12]
            };

            // Check:
            // beta01 + beta12 + beta02 = d1 + d2
            // = c1 + p2(alpha01) + p2(c2)
            // = p2(p1(b0)) + p2(alpha01) + p2(c2)
            // = p2(p1(b0) + alpha01 + c2)
            // = p2(p1(p0(x2) + alpha02) + c0)
            // = p2(p1(p0(x2) + alpha02) + p1(p0(x0 + x1)) - p1(alpha02))
            // = p2(p1(p0(x2) + alpha02 + p0(x0 + x1) - alpha02))
            // = p2(p1(p0(x2 + x0 + x1)))
            // = p(x)

            let o = g.create_tuple(shuffled)?;
            o.set_as_output()?;
        }
        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        format!(
            "ApplyPermutationMPC(inverse_permutation={}, reveal_output={})",
            self.inverse_permutation, self.reveal_output
        )
    }
}

impl ApplyPermutationMPC {
    fn apply_public_permutation(&self, input: Node, perm: Node, g: Graph) -> Result<Graph> {
        let n = perm.get_type()?.get_shape()[0];
        let perm = if self.inverse_permutation {
            perm.inverse_permutation()?
        } else {
            perm
        };
        if input.get_type()?.is_tuple() {
            // Array is secret-shared.
            if let Type::Tuple(shape) = input.get_type()? {
                if n != shape[0].get_shape()[0] {
                    return Err(runtime_error!(
                        "ApplyPermutationMPC: input and permutation should have the same first dimension."
                    ));
                }
                check_private_tuple(shape)?;
            } else {
                return Err(runtime_error!("Unreachable"));
            }
            let mut shares = vec![];
            for i in 0..PARTIES as u64 {
                let share = g.tuple_get(input.clone(), i)?;
                shares.push(apply_permutation(share, perm.clone())?);
            }
            let output = g.create_tuple(shares)?;
            if self.reveal_output {
                publish_node(output)?.set_as_output()?;
            } else {
                output.set_as_output()?;
            }
        } else {
            // Array is public.
            if n != input.get_type()?.get_shape()[0] {
                return Err(runtime_error!(
                    "ApplyPermutationMPC: input and permutation should have the same first dimension."
                ));
            }
            apply_permutation(input, perm)?.set_as_output()?;
        }
        g.finalize()
    }
}

fn apply_permutation(node: Node, perm: Node) -> Result<Node> {
    node.gather(perm, 0)
}

fn publish_node(a: Node) -> Result<Node> {
    // IF PARTIES=3: The only missing share for party i is the share with index i - 1.
    let mut shares = vec![];
    for i in 0..PARTIES {
        shares.push(a.tuple_get(i as u64)?);
        shares[i] = shares[i]
            .nop()?
            .add_annotation(NodeAnnotation::Send(i as u64, ((i + 1) % PARTIES) as u64))?;
    }
    let mut result = shares[0].clone();
    for share in shares.iter().skip(1) {
        result = result.add(share.clone())?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::custom_ops::{run_instantiation_pass, CustomOperation};
    use crate::data_types::{array_type, tuple_type, INT32, UINT64};
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::graphs::util::simple_context;
    use crate::inline::inline_ops::{inline_operations, InlineConfig, InlineMode};
    use crate::mpc::mpc_compiler::{generate_prf_key_triple, IOStatus};
    use crate::mpc::mpc_equivalence_class::{generate_equivalence_class, EquivalenceClasses};
    use crate::random::PRNG;
    use crate::typed_value::TypedValue;

    fn prepare_context(
        input_status: Vec<IOStatus>,
        output_parties: Vec<IOStatus>,
        t: Type,
        inline_config: InlineConfig,
        inverse_permutation: bool,
    ) -> Result<Context> {
        let c = simple_context(|g| {
            let reveal_output = !output_parties.is_empty();
            let custom_op = CustomOperation::new(ApplyPermutationMPC {
                inverse_permutation,
                reveal_output,
            });
            let p_t = array_type(vec![t.get_shape()[0]], UINT64);

            let i = if input_status[0] == IOStatus::Shared {
                g.input(tuple_type(vec![t.clone(); PARTIES]))?
            } else {
                g.input(t.clone())?
            };
            if input_status[1] == IOStatus::Shared {
                let p = g.input(tuple_type(vec![p_t.clone(); PARTIES]))?;
                let keys_vec = generate_prf_key_triple(g.clone())?;
                let keys = g.create_tuple(keys_vec)?;
                g.custom_op(custom_op, vec![i, p, keys])
            } else {
                let p = g.input(p_t.clone())?;
                g.custom_op(custom_op, vec![i, p])
            }
        })?;
        let instantiated_context = run_instantiation_pass(c)?.get_context();
        inline_operations(instantiated_context, inline_config)
    }

    fn prepare_input(input: TypedValue, input_status: IOStatus) -> Result<Value> {
        if input_status == IOStatus::Public || matches!(input_status, IOStatus::Party(_)) {
            return Ok(input.value);
        }

        let mut prng = PRNG::new(None)?;
        Ok(input.secret_share(&mut prng)?.value)
    }

    fn create_permutation(permutation: PermutationType) -> Result<Vec<u64>> {
        match permutation {
            PermutationType::Random(n) => {
                let mut prng = PRNG::new(None)?;
                let mut permutation = (0..n as u64).collect::<Vec<_>>();
                crate::evaluators::simple_evaluator::shuffle_array(&mut permutation, &mut prng)?;
                Ok(permutation)
            }
            PermutationType::Reverse(n) => Ok((0..n as u64).rev().collect()),
        }
    }

    fn prepare_permutation(
        permutation: PermutationType,
        input_status: IOStatus,
    ) -> Result<Vec<Vec<u64>>> {
        match input_status {
            IOStatus::Public => Ok(vec![create_permutation(permutation)?]),
            IOStatus::Party(_) => unimplemented!("Party input not supported"),
            IOStatus::Shared => Ok(vec![
                create_permutation(permutation.clone())?,
                create_permutation(permutation.clone())?,
                create_permutation(permutation)?,
            ]),
        }
    }

    fn convert_to_value(array: Vec<Vec<u64>>) -> Result<Value> {
        if array.len() == 1 {
            Value::from_flattened_array(&array[0], UINT64)
        } else if array.len() == 3 {
            Ok(Value::from_vector(vec![
                Value::from_flattened_array(&array[0], UINT64)?,
                Value::from_flattened_array(&array[1], UINT64)?,
                Value::from_flattened_array(&array[2], UINT64)?,
            ]))
        } else {
            unreachable!("Invalid number of shares")
        }
    }

    fn evaluate(
        mpc_graph: Graph,
        input_status: Vec<IOStatus>,
        input: TypedValue,
        permutation: PermutationType,
        output_parties: Vec<IOStatus>,
        inverse_permutation: bool,
    ) -> Result<()> {
        let permutations = prepare_permutation(permutation, input_status[1].clone())?;
        let output = random_evaluate(
            mpc_graph.clone(),
            vec![
                prepare_input(input.clone(), input_status[0].clone())?,
                convert_to_value(permutations.clone())?,
            ],
        )?;
        let t = input.t.clone();

        let output = if !output_parties.is_empty() {
            output.to_flattened_array_u128(t.clone())
        } else {
            // check that mpc_output is a sharing of plain_output
            assert!(output.check_type(tuple_type(vec![t.clone(); PARTIES]))?);
            // check that output is a sharing of expected
            output.access_vector(|v| match t.clone() {
                Type::Array(_, _) => {
                    let mut res = vec![0; t.get_dimensions().into_iter().product::<u64>() as usize];
                    for val in v {
                        let arr = val.to_flattened_array_u128(t.clone())?;
                        for i in 0..arr.len() {
                            res[i as usize] = u128::wrapping_add(res[i as usize], arr[i as usize]);
                        }
                    }
                    Ok(res)
                }
                _ => unreachable!(),
            })
        }?;
        let input = input.value.to_flattened_array_u128(t.clone())?;
        let m = match t.get_scalar_type().get_modulus() {
            Some(m) => m,
            None => 2u128.pow(64),
        };
        let (input, output) = (
            input.iter().map(|x| (x % m)).collect::<Vec<_>>(),
            output.iter().map(|x| (x % m)).collect::<Vec<_>>(),
        );
        assert_eq!(input.len(), output.len());
        let n = permutations[0].len();
        let perm = if permutations.len() == 1 {
            permutations[0].clone()
        } else {
            let mut res: Vec<u64> = (0..n).map(|i| i as u64).collect();
            // p = p2(p1(p0)), but to compose we actually need to traverse backwards.
            for p in permutations.iter().rev() {
                for i in 0..n {
                    res[i] = p[res[i] as usize];
                }
            }
            res
        };
        let perm = if inverse_permutation {
            let mut res = vec![0; n];
            for i in 0..n {
                res[perm[i] as usize] = i as u64;
            }
            res
        } else {
            perm
        };
        let chunk_size = input.len() / n;
        let input_array = input
            .chunks(chunk_size)
            .map(|x| x.to_vec())
            .collect::<Vec<_>>();
        let output_array = output
            .chunks(chunk_size)
            .map(|x| x.to_vec())
            .collect::<Vec<_>>();

        for i in 0..n {
            assert_eq!(input_array[perm[i] as usize], output_array[i]);
        }
        Ok(())
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    enum PermutationType {
        Random(usize),
        Reverse(usize),
    }

    #[test]
    fn test_correctness() -> Result<()> {
        let inline_config = InlineConfig {
            default_mode: InlineMode::Simple,
            ..Default::default()
        };
        let helper = |input: Vec<i32>,
                      permutation: PermutationType,
                      input_status: Vec<IOStatus>,
                      output_parties: Vec<IOStatus>,
                      inverse_permutation: bool,
                      is_2d: bool|
         -> Result<()> {
            let t = array_type(
                if is_2d {
                    vec![input.len() as u64, 2]
                } else {
                    vec![input.len() as u64]
                },
                INT32,
            );
            let mpc_context = prepare_context(
                input_status.clone(),
                output_parties.clone(),
                t.clone(),
                inline_config.clone(),
                inverse_permutation,
            )?;
            let mut advanced_input = vec![];
            for item in input {
                advanced_input.push(item);
                if is_2d {
                    // Check that for arrays [n, 2] it also works.
                    advanced_input.push(0);
                }
            }
            let input = TypedValue::new(
                t.clone(),
                Value::from_flattened_array(&advanced_input, t.get_scalar_type())?,
            )?;
            let mpc_graph = mpc_context.get_main_graph()?;

            evaluate(
                mpc_graph,
                input_status,
                input,
                permutation,
                output_parties,
                inverse_permutation,
            )?;

            Ok(())
        };
        let helper_nd = |input: Vec<i32>,
                         permutation: PermutationType,
                         input_status: Vec<IOStatus>,
                         output_parties: Vec<IOStatus>,
                         inverse_permutation: bool|
         -> Result<()> {
            helper(
                input.clone(),
                permutation.clone(),
                input_status.clone(),
                output_parties.clone(),
                inverse_permutation,
                false,
            )?;
            helper(
                input.clone(),
                permutation.clone(),
                input_status.clone(),
                output_parties.clone(),
                inverse_permutation,
                true,
            )?;

            Ok(())
        };
        let helper_inverse = |input: Vec<i32>,
                              permutation: PermutationType,
                              input_status: Vec<IOStatus>,
                              output_parties: Vec<IOStatus>|
         -> Result<()> {
            helper_nd(
                input.clone(),
                permutation.clone(),
                input_status.clone(),
                output_parties.clone(),
                false,
            )?;
            helper_nd(
                input.clone(),
                permutation.clone(),
                input_status.clone(),
                output_parties.clone(),
                true,
            )?;

            Ok(())
        };
        let helper_permutation_type = |input: Vec<i32>,
                                       input_status: Vec<IOStatus>,
                                       output_parties: Vec<IOStatus>|
         -> Result<()> {
            let n = input.len();
            helper_inverse(
                input.clone(),
                PermutationType::Random(n),
                input_status.clone(),
                output_parties.clone(),
            )?;
            // It doesn't make sense to test inverse permutation for the reverse permutation.
            helper_nd(
                input,
                PermutationType::Reverse(n),
                input_status,
                output_parties,
                false,
            )?;
            Ok(())
        };
        let helper_permutation_status = |input: Vec<i32>,
                                         input_status: IOStatus,
                                         output_parties: Vec<IOStatus>|
         -> Result<()> {
            helper_permutation_type(
                input.clone(),
                vec![input_status.clone(), IOStatus::Public],
                output_parties.clone(),
            )?;
            helper_permutation_type(
                input.clone(),
                vec![input_status.clone(), IOStatus::Shared],
                output_parties.clone(),
            )?;
            Ok(())
        };
        let helper_runs = |inputs: Vec<i32>| -> Result<()> {
            helper_permutation_type(
                inputs.clone(),
                vec![IOStatus::Public, IOStatus::Shared], // If everything is public, it should remain public.
                vec![],
            )?;
            helper_permutation_status(
                inputs.clone(),
                IOStatus::Public,
                vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
            )?;
            helper_permutation_status(inputs.clone(), IOStatus::Shared, vec![])?;
            helper_permutation_status(
                inputs.clone(),
                IOStatus::Shared,
                vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
            )?;
            Ok(())
        };
        helper_runs(vec![85])?;
        helper_runs(vec![10, 5])?;
        helper_runs(vec![3241532, 121, 1314, 19, -12, -13, i32::MAX, i32::MIN])?;
        helper_runs((-1000..1000).collect())?;
        Ok(())
    }

    #[test]
    fn test_classes_for_private_case() -> Result<()> {
        let inline_config = InlineConfig {
            default_mode: InlineMode::Simple,
            ..Default::default()
        };
        let input_status = vec![IOStatus::Shared, IOStatus::Shared];
        let output_parties = vec![];
        let mpc_context = prepare_context(
            input_status.clone(),
            output_parties.clone(),
            array_type(vec![2], INT32),
            inline_config.clone(),
            false,
        )?;

        let result_hashmap =
            generate_equivalence_class(mpc_context.clone(), vec![input_status.clone()])?;

        let share0_12 = EquivalenceClasses::Atomic(vec![vec![0], vec![1, 2]]);
        let share1_02 = EquivalenceClasses::Atomic(vec![vec![1], vec![0, 2]]);
        let share2_01 = EquivalenceClasses::Atomic(vec![vec![2], vec![0, 1]]);
        let shared = EquivalenceClasses::Vector(vec![
            Arc::new(share1_02.clone()),
            Arc::new(share2_01.clone()),
            Arc::new(share0_12.clone()),
        ]);
        let private = EquivalenceClasses::Atomic(vec![vec![0], vec![1], vec![2]]);

        let expected_classes = vec![
            // Secret shared input.
            shared.clone(),
            // Secret shared permutations.
            shared.clone(),
            // Create prf key.
            private.clone(),
            // Share it between parties 0 and 2.
            share1_02.clone(),
            // Create prf key.
            private.clone(),
            // Share it between parties 0 and 1.
            share2_01.clone(),
            // Create prf key.
            private.clone(),
            // Share it between parties 1 and 2.
            share0_12.clone(),
            // Create prf tuple,
            shared.clone(),
            // Extract input from tuples.
            share1_02.clone(),
            share2_01.clone(),
            share0_12.clone(),
            // Extract prf keys from tuples.
            share1_02.clone(),
            share2_01.clone(),
            share0_12.clone(),
            // Extract permutation shares from tuples.
            share1_02.clone(),
            share2_01.clone(),
            share0_12.clone(),
            // Finally the protocol itself.
            // 1. Parties 0 and 2 compute alpha02 = PRF(k02):
            share1_02.clone(),
            // 2. Party 2 computes b0 = p0(x2) + alpha02 and sends it to party 1:
            // p0(x2)
            private.clone(),
            // b0 = p0(x2) + alpha02
            private.clone(),
            // Send
            share0_12.clone(),
            // 3. Party 1 computes c1 = p2(p1(b0)):
            // p1(b0)
            private.clone(),
            // c1 = p2(p1(b0))
            private.clone(),
            // 4. Party 0 computes c0 = p1(p0(x0 + x1)) - p1(alpha02):
            // x0 + x1
            private.clone(),
            // p0(x0 + x1)
            private.clone(),
            // p1(p0(x0 + x1))
            private.clone(),
            // p1(alpha02)
            private.clone(),
            // c0 = p1(p0(x0 + x1)) - p1(alpha02)
            private.clone(),
            // 5. Parties 0 and 1 compute alpha01 = PRF(k01):
            share2_01.clone(),
            // 6. Party 0 computes c2 = c0 - alpha01 and sends it to party 2:
            // c2 = c0 - alpha01
            private.clone(),
            // Send
            share1_02.clone(),
            // 7. Party 1 computes d1 = c1 + p2(alpha01):
            // p2(alpha01)
            private.clone(),
            // d1 = c1 + p2(alpha01)
            private.clone(),
            // 8. Party 2 computes d2 = p2(c2):
            private.clone(),
            // 9. Parties 0 and 1 compute beta01 = PRF(k01):
            share2_01.clone(),
            // 10. Parties 0 and 2 compute beta02 = PRF(k02):
            share1_02.clone(),
            // 11. Party 1 computes t1 = d1 - beta01 and sends it to party 2:
            // t1 = d1 - beta01
            private.clone(),
            // Send
            share0_12.clone(),
            // 12. Party 2 computes t2 = d2 - beta02 and sends it to party 1:
            private.clone(),
            // Send
            share0_12.clone(),
            // 13. Parties 1 and 2 compute beta12 = t1 + t2:
            share0_12.clone(),
            // Create secret shared output.
            shared.clone(),
        ];

        for (i, classes) in expected_classes.iter().enumerate() {
            assert_eq!(result_hashmap[&(0, i as u64)], *classes, "i = {}", i);
        }

        let mpc_context = prepare_context(
            input_status.clone(),
            output_parties.clone(),
            array_type(vec![2], INT32),
            inline_config.clone(),
            true,
        )?;

        let result_hashmap =
            generate_equivalence_class(mpc_context.clone(), vec![input_status.clone()])?;

        // For inverse case classes should be the same except:
        // 1. There are 3 extra operations to apply inverse permutation.
        // 2. There should be swap: 0 <-> 1.

        // Skip first 18 steps as a preparation for the protocol.
        // Skip 3 steps for inverse permutation.
        for (i, classes) in expected_classes.into_iter().enumerate().skip(18) {
            let swapped_classes = swap_classes(classes);
            assert_eq!(
                result_hashmap[&(0, i as u64 + 3)],
                swapped_classes,
                "i = {}",
                i
            );
        }

        Ok(())
    }

    fn swap_classes(classes: EquivalenceClasses) -> EquivalenceClasses {
        let share0_12 = EquivalenceClasses::Atomic(vec![vec![0], vec![1, 2]]);
        let share1_02 = EquivalenceClasses::Atomic(vec![vec![1], vec![0, 2]]);
        if classes == share0_12 {
            return share1_02;
        }
        if classes == share1_02 {
            return share0_12;
        }
        return classes;
    }

    #[test]
    fn test_classes_for_reveal_case() -> Result<()> {
        let inline_config = InlineConfig {
            default_mode: InlineMode::Simple,
            ..Default::default()
        };
        let input_status = vec![IOStatus::Shared, IOStatus::Shared];
        let output_parties = vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)];
        let mpc_context = prepare_context(
            input_status.clone(),
            output_parties.clone(),
            array_type(vec![2], INT32),
            inline_config.clone(),
            false,
        )?;

        let result_hashmap =
            generate_equivalence_class(mpc_context.clone(), vec![input_status.clone()])?;

        let share0_12 = EquivalenceClasses::Atomic(vec![vec![0], vec![1, 2]]);
        let share1_02 = EquivalenceClasses::Atomic(vec![vec![1], vec![0, 2]]);
        let share2_01 = EquivalenceClasses::Atomic(vec![vec![2], vec![0, 1]]);
        let public = EquivalenceClasses::Atomic(vec![vec![0, 1, 2]]);
        let private = EquivalenceClasses::Atomic(vec![vec![0], vec![1], vec![2]]);

        let expected_classes = vec![
            // 1. Parties 0 and 2 compute alpha02 = PRF(k02):
            share1_02.clone(),
            // 2. Party 0 computes b1 = p0(x0 + x1) - alpha02 and sends it to party 1:
            // x0 + x1
            private.clone(),
            // p0(x0 + x1)
            private.clone(),
            // p0(x0 + x1) - alpha02
            private.clone(),
            // Send
            share2_01.clone(),
            // 3. Party 2 computes b2 = p0(x2) + alpha02 and sends it to party 1:
            // p0(x2)
            private.clone(),
            // b2 = p0(x2) + alpha02
            private.clone(),
            // Send
            share0_12.clone(),
            // 4. Party 1 computes c = p2(p1(b1 + b2)) and sends it to party 0 and 2.
            // b1 + b2
            private.clone(),
            // p1(b1 + b2)
            private.clone(),
            // p2(p1(b1 + b2))
            private.clone(),
            // Send
            share2_01.clone(),
            // Send
            public.clone(),
        ];

        // Skip first 18 steps as a preparation for the protocol.
        for (i, classes) in expected_classes.clone().into_iter().enumerate() {
            assert_eq!(result_hashmap[&(0, i as u64 + 18)], classes, "i = {}", i);
        }

        let mpc_context = prepare_context(
            input_status.clone(),
            output_parties.clone(),
            array_type(vec![2], INT32),
            inline_config.clone(),
            true,
        )?;

        let result_hashmap =
            generate_equivalence_class(mpc_context.clone(), vec![input_status.clone()])?;

        // For inverse case classes should be the same except:
        // 1. There are 3 extra operations to apply inverse permutation.
        // 2. There should be swap: 0 <-> 1.

        // Skip first 18 steps as a preparation for the protocol.
        // Skip 3 steps for inverse permutation.
        for (i, classes) in expected_classes.into_iter().enumerate() {
            let swapped_classes = swap_classes(classes);
            assert_eq!(
                result_hashmap[&(0, i as u64 + 21)],
                swapped_classes,
                "i = {}",
                i
            );
        }
        Ok(())
    }
}
