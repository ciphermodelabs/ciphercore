use crate::custom_ops::{CustomOperation, CustomOperationBody};
use crate::data_types::{array_type, get_types_vector, scalar_type, Type, BIT, UINT32};
use crate::data_values::Value;
use crate::errors::Result;
use crate::graphs::util::simple_context;
use crate::graphs::{Context, Graph, Node, Operation, SliceElement};
use crate::mpc::mpc_compiler::{check_private_tuple, PARTIES};
use crate::ops::utils::{custom_reduce, unsqueeze};

use serde::{Deserialize, Serialize};

use super::mpc_apply_permutation::ApplyPermutationMPC;
use super::utils::{convert_main_graph_to_mpc, get_column};

/// Implementation of stable RadixSort from <https://eprint.iacr.org/2019/695.pdf> (Algorithm 12).
/// * `key` - name of the column to be sorted.
/// * `bits_chunk_size` - number of bits to be processed together on the one counting step, aka `L` in section 5.3 "The optimized sorting protocol".
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub(super) struct RadixSortMPC {
    pub key: String,
    pub bits_chunk_size: u64,
}

/// According to the paper, the optimal value is 2 or 3. According to the benchmarks 2 is an optimal value.
const OPTIMAL_BITS_CHUNK_SIZE: u64 = 2;

impl RadixSortMPC {
    pub fn new(key: String) -> Self {
        Self {
            key,
            bits_chunk_size: OPTIMAL_BITS_CHUNK_SIZE,
        }
    }
}

/// RadixSort MPC operation for public and private data with the following arguments:
/// 1. named tuple with the column `self.key` of type `BIT` and shape `[n, b]` and other columns of arbitrary type and shape `[n, ...]`.
/// 2. PRF keys for MPC multiplication (only when data is private)
#[typetag::serde]
impl CustomOperationBody for RadixSortMPC {
    fn instantiate(&self, context: Context, argument_types: Vec<Type>) -> Result<Graph> {
        // If an input is private, i.e. a tuple of 3 elements (a0, a1, a2), then
        // the parties can access the following elements:
        // 1st party -> a0, a1;
        // 2nd party -> a1, a2;
        // 3rd party -> a2, a0.
        if argument_types.len() == 1 {
            let g = context.create_graph()?;
            let input = g.input(argument_types[0].clone())?;
            g.sort(input, self.key.clone())?.set_as_output()?;
            return g.finalize();
        }

        if let (Type::Tuple(v0), Type::Tuple(v1)) =
            (argument_types[0].clone(), argument_types[1].clone())
        {
            check_private_tuple(v0)?;
            check_private_tuple(v1)?;
        } else {
            return Err(runtime_error!(
                "RadixSortMPC should have a private tuple of input and a tuple of PRF keys as input"
            ));
        }
        let types = get_types_vector(argument_types[0].clone())?[0].get_named_types()?;
        let mut key_shape = vec![];
        for (name, t) in types.clone() {
            if name == self.key {
                key_shape = t.get_shape();
                break;
            }
        }
        let names = types
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<String>>();

        let (n, b) = (key_shape[0], key_shape[1]);

        // Compiler requires to generate this graphs before creating main graph.
        let gen_multi_bit_sort =
            gen_multi_bit_sort_graph(context.clone(), n, self.bits_chunk_size)?;
        let step0_size = b % self.bits_chunk_size;
        let (gen_multi_bit_sort_step0, step0_size) = if step0_size == 0 {
            (gen_multi_bit_sort.clone(), self.bits_chunk_size)
        } else {
            (
                gen_multi_bit_sort_graph(context.clone(), n, step0_size)?,
                step0_size,
            )
        };

        let g = context.create_graph()?;
        let arrays = g.input(argument_types[0].clone())?;
        let mut arrays_shares = vec![];
        for i in 0..PARTIES {
            arrays_shares.push(arrays.tuple_get(i as u64)?);
        }
        let arrays = names
            .iter()
            .map(|name| get_column(&arrays_shares, name.clone()))
            .collect::<Result<Vec<Node>>>()?;

        let input = permute_axes_mpc(get_column(&arrays_shares, self.key.clone())?, vec![1, 0])?;
        let prf_keys = g.input(argument_types[1].clone())?;
        let k = get_slice_mpc(
            input.clone(),
            vec![SliceElement::SubArray(
                Some(-(step0_size as i64)),
                None,
                None,
            )],
        )?;
        // Algorithm 12: Optimized permutation generation of stable sort.
        let mut sigma = g.call(gen_multi_bit_sort_step0, vec![prf_keys.clone(), k])?;
        let bit_chunks_count = (b - step0_size) / self.bits_chunk_size;
        let input = if bit_chunks_count != 0 {
            let input = get_slice_mpc(
                input,
                vec![SliceElement::SubArray(
                    None,
                    Some(-(step0_size as i64)),
                    None,
                )],
            )?; // Cut off the last `step0_size` bits.
            let input = reshape_mpc(
                input,
                array_type(vec![bit_chunks_count, self.bits_chunk_size, n], BIT),
            )?; // [bit_chunks_count, bits_chunk_size, n]
            permute_axes_mpc(input, vec![0, 2, 1])? // [bit_chunks_count, n, bits_chunk_size]
        } else {
            input
        };
        // Iterating from least significant to most significant bits.
        for bit_ind in (0..bit_chunks_count).rev() {
            let input_chunk = get_mpc(input.clone(), vec![bit_ind])?;
            // Algorithm 4: Applying the inverse of a share-vector permutation.
            let pi = secret_shared_permutation(prf_keys.clone(), n)?;
            let k = shuffle(input_chunk, pi.clone(), prf_keys.clone())?;
            sigma = shuffle_and_reveal(sigma, pi.clone(), prf_keys.clone())?;
            let k = apply_permutation_plaintext(k, sigma.clone(), true)?;
            // Algorithm 11: Generating permutation of stable sort for multiple bits.
            let ro = g.call(
                gen_multi_bit_sort.clone(),
                vec![prf_keys.clone(), permute_axes_mpc(k, vec![1, 0])?],
            )?;
            // Algorithm 14: Optimized composition of two permutations.
            sigma = apply_permutation_plaintext(ro, sigma, false)?;
            sigma = unshuffle(sigma, pi, prf_keys.clone())?;
        }

        let result = apply_sorting_permutation(arrays, sigma, prf_keys, n)?;
        create_named_tuple_mpc(g.clone(), result, names)?.set_as_output()?;
        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        format!(
            "RadixSortMPC(key={}, bits_group_size={})",
            self.key, self.bits_chunk_size
        )
    }
}

// Algorithm 13: Optimized inverse application of a permutation.
fn apply_sorting_permutation(
    nodes: Vec<Node>,
    sigma: Node,
    prf_keys: Node,
    n: u64,
) -> Result<Vec<Node>> {
    let pi = secret_shared_permutation(prf_keys.clone(), n)?;
    let sigma = shuffle_and_reveal(sigma, pi.clone(), prf_keys.clone())?;
    let mut result = vec![];
    for node in nodes {
        let shuffled_node = shuffle(node, pi.clone(), prf_keys.clone())?;
        result.push(apply_permutation_plaintext(
            shuffled_node,
            sigma.clone(),
            true,
        )?)
    }
    Ok(result)
}

// Apply public permutation to the secret-shared input.
fn apply_permutation_plaintext(input: Node, p: Node, inverse_permutation: bool) -> Result<Node> {
    let g = input.get_graph();
    g.custom_op(
        CustomOperation::new(ApplyPermutationMPC {
            inverse_permutation,
            reveal_output: false,
        }),
        vec![input, p],
    )
}

fn shuffle(node: Node, p: Node, prf_keys: Node) -> Result<Node> {
    let g = node.get_graph();
    g.custom_op(
        CustomOperation::new(ApplyPermutationMPC {
            inverse_permutation: false,
            reveal_output: false,
        }),
        vec![node, p, prf_keys],
    )
}

fn shuffle_and_reveal(node: Node, p: Node, prf_keys: Node) -> Result<Node> {
    let g = node.get_graph();
    g.custom_op(
        CustomOperation::new(ApplyPermutationMPC {
            inverse_permutation: false,
            reveal_output: true,
        }),
        vec![node, p, prf_keys],
    )
}

fn unshuffle(node: Node, p: Node, prf_keys: Node) -> Result<Node> {
    let g = node.get_graph();
    g.custom_op(
        CustomOperation::new(ApplyPermutationMPC {
            inverse_permutation: true,
            reveal_output: false,
        }),
        vec![node, p, prf_keys],
    )
}

fn secret_shared_permutation(prf_keys: Node, n: u64) -> Result<Node> {
    generic_function_on_shares(prf_keys, Operation::PermutationFromPRF(0, n))
}

fn get_slice_mpc(node: Node, slice: Vec<SliceElement>) -> Result<Node> {
    generic_function_on_shares(node, Operation::GetSlice(slice))
}

fn create_named_tuple_mpc(graph: Graph, nodes: Vec<Node>, names: Vec<String>) -> Result<Node> {
    let mut shares = vec![];
    for i in 0..PARTIES {
        let mut share = vec![];
        for (name, node) in names.iter().zip(nodes.iter()) {
            share.push((name.clone(), node.tuple_get(i as u64)?));
        }
        shares.push(graph.create_named_tuple(share)?);
    }
    graph.create_tuple(shares)
}

fn permute_axes_mpc(node: Node, axes: Vec<u64>) -> Result<Node> {
    generic_function_on_shares(node, Operation::PermuteAxes(axes))
}

fn reshape_mpc(node: Node, t: Type) -> Result<Node> {
    generic_function_on_shares(node, Operation::Reshape(t))
}

fn get_mpc(node: Node, indices: Vec<u64>) -> Result<Node> {
    generic_function_on_shares(node, Operation::Get(indices))
}

fn generic_function_on_shares(node: Node, op: Operation) -> Result<Node> {
    let g = node.get_graph();
    let mut shares = vec![];
    for i in 0..PARTIES {
        shares.push(g.add_node(vec![node.tuple_get(i as u64)?], vec![], op.clone())?);
    }
    g.create_tuple(shares)
}

/// Algorithm 11: Generating permutation of stable sort for multiple bits.
/// Let's take a closer look how it would work for n = 3 and l = 2.
/// Let input be: input = [2, 0, 3]
/// We expect output to be: [1, 0, 2] (inversed sorting permutation in 0-indexation)
/// Input in bit-representation:
/// [[1, 0, 1],
/// [0, 0, 1]]
/// xor_mask (just inverted numbers for 0..2^l):
/// [[1, 1, 0, 0],
///  [1, 0, 1, 0]]
/// d:
/// [[
/// [0, 1, 0],
/// [0, 1, 0],
/// [1, 0, 1],
/// [1, 0, 1]],
/// [
/// [1, 1, 0],
/// [0, 0, 1],
/// [1, 1, 0],
/// [0, 0, 1]]]
/// f (permuted axes: [n, num_values]) is one-hot encoding of the input:
/// [[0, 0, 1, 0], // first number in input is 2
///  [1, 0, 0, 0], // second number in input is 0
///  [0, 0, 0, 1]] // third number in input is 3
/// f_prefix_sum:
/// [[0, 0, 1, 0],
///  [1, 0, 1, 0],
///  [1, 0, 1, 1]]
/// last_row_prefix_sum (this is non inclusive prefix sum):
/// [0, 1, 1, 2]
/// s:
/// [[0, 1, 2, 2],
///  [1, 1, 2, 2],
///  [1, 1, 2, 3]]
/// s * f:
/// [[0, 0, 2, 0],
///  [1, 0, 0, 0],
///  [0, 0, 0, 3]]
/// In each row we have exactly one non-zero element. (because f is one-hot and also had exactly one non-zero element in each row).
/// p: [2, 1, 3] - permutation of input in 1-indexation.
/// p: [1, 0, 2] - permutation of input in 0-indexation (this is the output we expected)
fn gen_multi_bit_sort_graph(context: Context, n: u64, l: u64) -> Result<Graph> {
    let gen_bit_perm_context = simple_context(|g| {
        let t = array_type(vec![l, n], BIT);
        let input = g.input(t)?; // [l, n]
        let num_values = 2_usize.pow(l as u32);
        let mut const_bits = vec![];
        // Reverse order to get the correct order of bits.
        for k in (0..l as u32).rev() {
            for j in 0..num_values {
                if j & (1 << k) == 0 {
                    const_bits.push(1);
                } else {
                    const_bits.push(0);
                }
            }
        }
        // Let x[i] be the number corresponding to input[..][i].
        // For each value v from 0..num_values we precompute the xor_mask[v]
        // xor_mask[v] is bitstring that we need to xor the input bitstring x[i] with, such that we get a bitstring of all-ones if x[i] == v.
        let xor_mask = g.constant(
            array_type(vec![l, num_values as u64], BIT),
            Value::from_flattened_array(&const_bits, BIT)?,
        )?; // [l, num_values]
        let d = unsqueeze(input, -2)?.add(unsqueeze(xor_mask, -1)?)?; // [l, num_values, n]

        // f is one_hot encoding of x[i] in the sense that:
        // f[v][i] = 1 if (x[i] == v) else 0
        let f = custom_reduce(d, |first, second| first.multiply(second))?; // [num_values, n]
        let one = g.ones(scalar_type(UINT32))?;
        let f = one.mixed_multiply(f.permute_axes(vec![1, 0])?)?; // [n, num_values]

        // Here we are going to compute s:
        // s is the cumulative sum of f from left to right and from top to bottom.
        // s[i][v] = sum(f[j][u]; j <= i, u <= v).
        let f_prefix_sum = f.cum_sum(0)?;

        let last_row_prefix_sum = pad_left(
            f_prefix_sum
                .get(vec![n - 1])? // lasе row
                .cum_sum(0)? // cumsum
                .get_slice(vec![SliceElement::SubArray(None, Some(-1), None)])?, // remove last element
            1,
        )?; // [num_values]

        let s = unsqueeze(last_row_prefix_sum, 0)?.add(f_prefix_sum)?; // [n, num_values]

        let p = custom_reduce(s.multiply(f)?.permute_axes(vec![1, 0])?, |first, second| {
            first.add(second)
        })?; // [n]
        p.subtract(one) // 0-indexation
    })?;
    convert_main_graph_to_mpc(gen_bit_perm_context, context, vec![true])
}

fn pad_left(data: Node, pad_size: u64) -> Result<Node> {
    let (shape, sc) = match data.get_type()? {
        Type::Array(shape, sc) => (shape, sc),
        _ => return Err(runtime_error!("Expected array type")),
    };
    let mut pad_shape = shape;
    pad_shape[0] = pad_size;
    let g = data.get_graph();
    let pad = g.zeros(array_type(pad_shape, sc))?;
    g.concatenate(vec![pad, data], 0)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::bytes::add_vectors_u128;
    use crate::data_types::{array_type, tuple_type};
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::graphs::util::simple_context;
    use crate::inline::inline_ops::{InlineConfig, InlineMode};
    use crate::mpc::mpc_compiler::{prepare_for_mpc_evaluation, IOStatus};
    use crate::random::PRNG;
    use crate::typed_value::TypedValue;

    fn prepare_context(
        input_status: Vec<IOStatus>,
        output_parties: Vec<IOStatus>,
        t: Type,
        inline_config: InlineConfig,
    ) -> Result<Context> {
        let c = simple_context(|g| {
            let i = g.input(t.clone())?;
            let key = "key".to_string();
            let t = g.create_named_tuple(vec![(key.clone(), i)])?;
            g.sort(t, key.clone())?.named_tuple_get(key)
        })?;
        Ok(
            prepare_for_mpc_evaluation(c, vec![input_status], vec![output_parties], inline_config)?
                .get_context(),
        )
    }

    fn prepare_input(input: TypedValue, input_status: IOStatus) -> Result<Value> {
        if input_status == IOStatus::Public || matches!(input_status, IOStatus::Party(_)) {
            return Ok(input.value);
        }

        let mut prng = PRNG::new(None)?;
        Ok(input.secret_share(&mut prng)?.value)
    }

    fn evaluate(
        mpc_graph: Graph,
        input_status: Vec<IOStatus>,
        input: TypedValue,
        output_parties: Vec<IOStatus>,
    ) -> Result<()> {
        let output = random_evaluate(
            mpc_graph.clone(),
            vec![prepare_input(input.clone(), input_status[0].clone())?],
        )?;
        let t = input.t.clone();
        let output = if !output_parties.is_empty() {
            output.to_flattened_array_u128(t.clone())
        } else {
            // check that mpc_output is a sharing of plain_output
            assert!(output.check_type(tuple_type(vec![t.clone(); PARTIES]))?);
            // check that output is a sharing of expected
            output.access_vector(|v| match t.clone() {
                Type::Array(_, st) => {
                    let mut res = vec![0; t.get_dimensions().into_iter().product::<u64>() as usize];
                    for val in v {
                        let arr = val.to_flattened_array_u128(t.clone())?;
                        res = add_vectors_u128(&res, &arr, st.get_modulus())?;
                    }
                    Ok(res)
                }
                _ => unreachable!(),
            })
        }?;
        let n = t.get_shape()[0];
        let input = input.value.to_flattened_array_u128(t.clone())?;
        let mut input = input
            .chunks(input.len() / n as usize)
            .map(|x| x.to_vec())
            .collect::<Vec<_>>();
        let output = output
            .chunks(output.len() / n as usize)
            .map(|x| x.to_vec())
            .collect::<Vec<_>>();
        input.sort();
        assert_eq!(input, output);

        Ok(())
    }

    #[test]
    fn test_correctness() -> Result<()> {
        let inline_config = InlineConfig {
            default_mode: InlineMode::Simple,
            ..Default::default()
        };
        let helper = |input: TypedValue,
                      input_status: Vec<IOStatus>,
                      output_parties: Vec<IOStatus>|
         -> Result<()> {
            let mpc_context = prepare_context(
                input_status.clone(),
                output_parties.clone(),
                input.t.clone(),
                inline_config.clone(),
            )?;
            let mpc_graph = mpc_context.get_main_graph()?;

            evaluate(mpc_graph, input_status, input, output_parties)?;

            Ok(())
        };
        let helper_runs = |input: TypedValue| -> Result<()> {
            helper(input.clone(), vec![IOStatus::Public], vec![])?;
            helper(
                input.clone(),
                vec![IOStatus::Public],
                vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
            )?;
            helper(input.clone(), vec![IOStatus::Shared], vec![])?;
            helper(
                input.clone(),
                vec![IOStatus::Shared],
                vec![IOStatus::Party(0), IOStatus::Party(1), IOStatus::Party(2)],
            )?;
            Ok(())
        };
        let input = Value::from_flattened_array(&[1, 0, 1, 0, 0, 0, 1], BIT)?;
        helper_runs(TypedValue::new(array_type(vec![7, 1], BIT), input)?)?;
        let input = Value::from_flattened_array(&[1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0], BIT)?;
        helper_runs(TypedValue::new(array_type(vec![7, 2], BIT), input)?)?;
        let mut prng = PRNG::new(None)?;
        let input = Value::from_flattened_array(
            &(0..(200 * 32))
                .map(|_| prng.get_random_in_range(Some(2)))
                .collect::<Result<Vec<_>>>()?,
            BIT,
        )?;
        helper_runs(TypedValue::new(array_type(vec![200, 32], BIT), input)?)?;
        Ok(())
    }
}
