use crate::broadcast::number_to_index;
use crate::data_types::{array_type, scalar_type, Type, BIT, UINT64};
use crate::data_values::Value;
use crate::errors::Result;
use crate::graphs::{Graph, Node, SliceElement};
use crate::inline::data_structures::{log_depth_sum, CombineOp};
use crate::inline::inline_common::{
    pick_prefix_sum_algorithm, DepthOptimizationLevel, InlineState,
};
use crate::ops::utils::constant_scalar;

const MAX_ALLOWED_STATE_BITS: u64 = 4;

/// This version inlines Iterate assuming the state has low number of bits.
/// If there are K bits, the additional time complexity multiplier is 2**(K+2),
/// and the resulting depth is O(log(length of the inputs)).
/// Important note about the contract: state can be batched, it is assumed that
/// the last dimension of the state is the actual state consisting of K bits, and
/// there are no interactions between various states.
/// More formally:
/// -- let the input state S have shape (..., K) (rank d);
/// -- let T = Call(graph, S, some input);
/// -- let i_1, .., i_{d-1} be a valid combination of indices,
/// -- then for every S' such that
///    S[i_1, .., i_{d-1}] == S'[i_1, .., i_{d-1}] (as vectors),
///    the result T[i_1, .., i_{d-1}] equals T'[i_1, .., i_{d-1}].
/// (i.e. we can change other "rows" of the input, and the output for this "row"
/// won't change).
/// If this property is not true, results might be incorrect.
///
/// There is also a special case of K == 1, where the caller can (but not must)
/// omit the last dimension in the state (e.g. have state of shape (n, m) rather
/// than (n, m, 1)).
pub(super) fn inline_iterate_small_state(
    single_bit: bool,
    optimization_level: DepthOptimizationLevel,
    graph: Graph,
    initial_state: Node,
    inputs_node: Node,
    inliner: &mut dyn InlineState,
) -> Result<(Node, Vec<Node>)> {
    // TODO(ilyakor): for "batched state" case, support the "all items are equal" optimization.
    let graph_output_type = graph.get_output_node()?.get_type()?;
    let output_element_type = match graph_output_type {
        Type::Tuple(tuple_types) => (*tuple_types[1]).clone(),
        _ => {
            panic!("Inconsistency with type checker for Iterate output.");
        }
    };
    let empty_output = match output_element_type {
        Type::Tuple(tuple_types) => tuple_types.is_empty(),
        _ => false,
    };

    let inputs_len = match inputs_node.get_type()? {
        Type::Vector(len, _) => len,
        _ => {
            panic!("Inconsistency with type checker");
        }
    };
    if inputs_len == 0 {
        return Ok((initial_state, vec![]));
    }

    let num_bits = get_number_of_bits(initial_state.get_type()?, single_bit)?;
    if num_bits > MAX_ALLOWED_STATE_BITS {
        return Err(runtime_error!("Too many bits in the state"));
    }
    if num_bits == 0 {
        return Err(runtime_error!(
            "This inlining method doesn't support empty state"
        ));
    }
    let num_masks = u64::pow(2, num_bits as u32);

    // Intuitively, the algorithm is as follows.
    // Let's look at our operation G(state, input) -> state'. In general, it can be
    // non-associative, however, we can do the following trick. Let's compute
    // matrices M_input of form M[state1, state2] = (G(state1, input) == state2).
    // One can see that the resulting state is:
    //   OneHot(initial_state) * M(input1) * M(input2) * ....
    // Where "*" is MatMul. But matrix multiplication is associative, so we can use the
    // same tricks as the associative case to reduce the depth.
    //
    // While simple conceptually, the code is quite complicated since we want to handle
    // states of shape (..., K bits), not just (K bits).

    // Precalculate K-bit masks, they're used in many places.
    let state_type = initial_state.get_type()?;
    let mut mask_constants = vec![];
    for mask in 0..u64::pow(2, num_bits as u32) {
        let value = mask_to_value(state_type.clone(), num_bits, mask)?;
        let mask_const = inliner.output_graph().constant(state_type.clone(), value)?;
        mask_constants.push(mask_const);
    }

    // First, let's compute our mappings in form of matrices.
    // If state shape is (..., K), the shape of mappings is (..., K, K).
    let mappings = create_mappings(
        initial_state.get_type()?,
        mask_constants.clone(),
        num_bits,
        single_bit,
        inputs_node.clone(),
        graph.clone(),
        inliner,
    )?;

    // Precompute the transformation of initial_state, which is needed to
    // extract states from the transformation matrices. See extract_state_from_mapping()
    // for more detailed explanation.
    let unused_node = inliner.output_graph().zeros(scalar_type(BIT))?;
    let initial_state_one_hot = if single_bit {
        unused_node.clone()
    } else {
        let mut initial_state_one_hot = one_hot_encode(
            initial_state.clone(),
            num_masks,
            mask_constants.clone(),
            inliner.output_graph(),
            state_type.clone(),
            single_bit,
        )?;
        let mut new_shape = initial_state_one_hot.get_type()?.get_shape();
        new_shape.insert(0, 1);
        initial_state_one_hot =
            initial_state_one_hot.reshape(array_type(new_shape.clone(), BIT))?;
        let mut permutation: Vec<u64> = (0..new_shape.len()).map(|x| x as u64).collect();
        permutation.rotate_left(2);
        initial_state_one_hot = initial_state_one_hot.permute_axes(permutation)?; // ...1i
        initial_state_one_hot
    };

    // Precompute the mask array, needed for extracting states. See extract_state_from_mapping()
    // for more detailed explanation.
    let masks_arr = if single_bit {
        unused_node
    } else {
        let masks_arr = inliner
            .output_graph()
            .create_vector(mask_constants[0].get_type()?, mask_constants)?
            .vector_to_array()?;
        let masks_arr_shape = masks_arr.get_type()?.get_shape();
        let mut masks_arr_permutation: Vec<u64> =
            (0..masks_arr_shape.len()).map(|x| x as u64).collect();
        masks_arr_permutation.rotate_left(1);
        let rank = masks_arr_permutation.len();
        masks_arr_permutation.swap(rank - 2, rank - 1);
        masks_arr.permute_axes(masks_arr_permutation)?
    };

    let mut combiner = MappingCombiner {};
    let mut bit_combiner = MappingCombiner1Bit {};
    if empty_output {
        // Outputs for this case are trivial.
        let mut outputs = vec![];
        let empty_tuple = inliner.output_graph().create_tuple(vec![])?;
        for _ in 0..inputs_len {
            outputs.push(empty_tuple.clone());
        }

        let final_mapping = if single_bit {
            log_depth_sum(&mappings, &mut bit_combiner)?
        } else {
            log_depth_sum(&mappings, &mut combiner)?
        };
        // We have the final mapping, let's compute and extract the answer.

        let result = extract_state_from_mapping(
            single_bit,
            initial_state,
            initial_state_one_hot,
            final_mapping,
            masks_arr,
            state_type,
        )?;
        Ok((result, outputs))
    } else {
        let prefix_sums = if single_bit {
            pick_prefix_sum_algorithm(inputs_len, optimization_level)(&mappings, &mut bit_combiner)?
        } else {
            pick_prefix_sum_algorithm(inputs_len, optimization_level)(&mappings, &mut combiner)?
        };
        let mut outputs = vec![];
        for i in 0..inputs_len {
            let state = if i == 0 {
                initial_state.clone()
            } else {
                extract_state_from_mapping(
                    single_bit,
                    initial_state.clone(),
                    initial_state_one_hot.clone(),
                    prefix_sums[i as usize - 1].clone(),
                    masks_arr.clone(),
                    state_type.clone(),
                )?
            };
            let input =
                inputs_node.vector_get(constant_scalar(&inliner.output_graph(), i, UINT64)?)?;
            inliner.assign_input_nodes(graph.clone(), vec![state, input])?;
            let output = inliner.recursively_inline_graph(graph.clone())?;
            inliner.unassign_nodes(graph.clone())?;
            outputs.push(output.tuple_get(1)?);
        }
        let result = extract_state_from_mapping(
            single_bit,
            initial_state,
            initial_state_one_hot,
            prefix_sums[prefix_sums.len() - 1].clone(),
            masks_arr,
            state_type,
        )?;
        Ok((result, outputs))
    }
}

struct MappingCombiner {}

impl CombineOp<Node> for MappingCombiner {
    fn combine(&mut self, arg1: Node, arg2: Node) -> Result<Node> {
        arg1.matmul(arg2)
    }
}

// Optimized version of mapping combiner for the single-bit case.
// It produces more operations, but the operations are more lightweight.
struct MappingCombiner1Bit {}

impl CombineOp<Node> for MappingCombiner1Bit {
    fn combine(&mut self, arg1: Node, arg2: Node) -> Result<Node> {
        // Mappings are in the form tuple(bit_0, bit_1), so we extract
        // both bits from the 2nd mapping, and combine mappings as follows:
        // output_0 = bit_10 * (bit20 + bit21) + bit_20;
        // output_1 = bit_11 * (bit20 + bit21) + bit_20.
        let bit10 = arg1.tuple_get(0)?;
        let bit11 = arg1.tuple_get(1)?;
        let bit20 = arg2.tuple_get(0)?;
        let bit21 = arg2.tuple_get(1)?;
        let distinct = bit20.add(bit21)?;
        let bit0 = bit10.multiply(distinct.clone())?.add(bit20.clone())?;
        let bit1 = bit11.multiply(distinct)?.add(bit20)?;
        arg1.get_graph().create_tuple(vec![bit0, bit1])
    }
}

fn extract_state_from_mapping(
    single_bit: bool,
    initial_state: Node,
    initial_state_one_hot: Node,
    mapping: Node,
    masks_arr: Node,
    state_type: Type,
) -> Result<Node> {
    if single_bit {
        // Optimized 1-bit case. The mapping is a tuple with 2 bits (0->bit_0, 1->bit_1).
        let g = mapping.get_graph();
        let out0 = mapping.tuple_get(0)?;
        let out1 = mapping.tuple_get(1)?;
        let one = g.ones(scalar_type(BIT))?;
        let not_initial_state = initial_state.add(one)?;
        out0.multiply(not_initial_state)?
            .add(out1.multiply(initial_state)?)
    } else {
        // Currently, initial_state_one_hot is "...i1", and final_mapping - "...ij".
        // We want to multiply them, but for this, we'll need Reshape and PermuteAxes.
        //
        // To make it easier to follow, here is what is happening below.
        // Let M (...ij) be the final mapping, S (...i) be the one-hot-encoded state, C (2**k, ..., k)
        // be the precomputed mask array.
        // O := einsum('...ij,...i->...j', M, S), this is the one-hot-encoded output state.
        // Result := einsum('...j,j...k->...k', M, C)
        // Then we just reshape the result to the correct shape.

        let output_state_one_hot = initial_state_one_hot.matmul(mapping)?;
        // Now we have one-hot encoded result, we just need to decode it.
        let final_state = output_state_one_hot.matmul(masks_arr)?;
        final_state.reshape(state_type)
    }
}

fn get_number_of_bits(state_type: Type, single_bit: bool) -> Result<u64> {
    match state_type {
        Type::Scalar(scalar_type) => {
            if !single_bit {
                Err(runtime_error!(
                    "Scalar state is only supported in a single-bit mode"
                ))
            } else if scalar_type != BIT {
                Err(runtime_error!("State must consist of bits"))
            } else {
                Ok(1)
            }
        }
        Type::Array(shape, scalar_type) => {
            if scalar_type != BIT {
                Err(runtime_error!("State must consist of bits"))
            } else if single_bit {
                Ok(1)
            } else {
                Ok(shape[shape.len() - 1])
            }
        }
        _ => Err(runtime_error!("Unsupported state type")),
    }
}

fn mask_to_value(state_type: Type, num_bits: u64, mask: u64) -> Result<Value> {
    let data_shape = match state_type.clone() {
        Type::Scalar(scalar_type) => {
            return Value::from_scalar(mask, scalar_type);
        }
        Type::Array(shape, _) => shape,
        _ => panic!("Cannot be here"),
    };
    let value = Value::zero_of_type(state_type);
    let mut bytes = value.access_bytes(|ref_bytes| Ok(ref_bytes.to_vec()))?;
    for i in 0..data_shape.iter().product() {
        let index = number_to_index(i, &data_shape);
        let state_index = if num_bits == 1 {
            0
        } else {
            index[index.len() - 1]
        };
        let bit = ((mask >> state_index) & 1) as u8;
        let position = i / 8;
        let offset = i % 8;
        bytes[position as usize] &= !(1 << offset);
        bytes[position as usize] |= bit << offset;
    }
    Ok(Value::from_bytes(bytes))
}

fn one_hot_encode(
    val: Node,
    depth: u64,
    mask_constants: Vec<Node>,
    output: Graph,
    state_type: Type,
    single_bit: bool,
) -> Result<Node> {
    let mut result = vec![];
    // We have a single (batched) value `val`, which we want to one-hot encode into a vector
    // of size `depth`.
    // We do this by comparing `val` with every single value in range [0, depth).
    // Each comparison is done by taking bit_diff := val xor ~target_val, which consists of
    // ones if there is equality. So we're multiplying bit_diff[..., i] for all i to check this.
    for mask in 0..depth {
        // We use ~mask to avoid taking negation within the graph.
        let column_id = mask_constants[((depth - 1) ^ mask) as usize].clone();
        let bit_diff = val.add(column_id)?;
        if single_bit {
            result.push(bit_diff.clone());
        } else {
            let shape = match state_type.clone() {
                Type::Array(shape, _) => shape,
                _ => panic!("Cannot be here"),
            };
            let mut bit_columns = vec![];
            for bit_index in 0..shape[shape.len() - 1] {
                bit_columns.push(bit_diff.get_slice(vec![
                    SliceElement::Ellipsis,
                    SliceElement::SingleIndex(bit_index as i64),
                ])?);
            }
            // Note: this can also be done with smaller depth (log(k) instead of k), we can
            // do it if it even becomes a problem.
            let mut equality = bit_columns[0].clone();
            for bit_index in 1..shape[shape.len() - 1] {
                equality = equality.multiply(bit_columns[bit_index as usize].clone())?;
            }
            result.push(equality.clone());
        }
    }

    output.vector_to_array(output.create_vector(result[0].get_type()?, result)?)
}

fn create_mapping_matrix(
    mapping: Vec<Node>,
    output: Graph,
    mask_constants: Vec<Node>,
    state_type: Type,
    single_bit: bool,
) -> Result<Node> {
    if single_bit {
        // Single-bit optimization: we don't need to one-hot encode the mapping in this case.
        return output.create_tuple(mapping);
    }
    // We're given 2 ** K mappings, where each mapping is a state. We want to produce
    // the transition matrix of shape (2 ** K, 2 ** K), by one-hot-encoding every state.
    // We want the transition matrix to be in the last two dimensions, so we do
    // PermuteAxes at the end of the process.
    let mut result = vec![];
    let depth = mapping.len() as u64;
    for node_to_map in mapping {
        result.push(one_hot_encode(
            node_to_map,
            depth,
            mask_constants.clone(),
            output.clone(),
            state_type.clone(),
            single_bit,
        )?);
    }
    let matrix = output.vector_to_array(output.create_vector(result[0].get_type()?, result)?)?;
    Ok(matrix)
}

/// Creates mappings and one-hot-encoded initial state.  
fn create_mappings(
    state_type: Type,
    mask_constants: Vec<Node>,
    num_bits: u64,
    single_bit: bool,
    inputs_node: Node,
    graph: Graph,
    inliner: &mut dyn InlineState,
) -> Result<Vec<Node>> {
    let inputs_len = match inputs_node.get_type()? {
        Type::Vector(len, _) => len,
        _ => {
            panic!("Inconsistency with type checker");
        }
    };
    let mut mappings = vec![];
    for i in 0..inputs_len {
        let current_input = inputs_node.vector_get(
            inliner
                .output_graph()
                .constant(scalar_type(UINT64), Value::from_scalar(i, UINT64)?)?,
        )?;
        let mut mapping_table = vec![];
        for mask in 0..u64::pow(2, num_bits as u32) {
            let current_state = mask_constants[mask as usize].clone();
            inliner.assign_input_nodes(
                graph.clone(),
                vec![current_state.clone(), current_input.clone()],
            )?;
            let output = inliner.recursively_inline_graph(graph.clone())?;
            inliner.unassign_nodes(graph.clone())?;
            mapping_table.push(inliner.output_graph().tuple_get(output, 0)?);
        }
        // Note: from the perspective of graph size, it might be a good idea to batch
        // matrix creation outsize of the loop.
        mappings.push(create_mapping_matrix(
            mapping_table,
            inliner.output_graph().clone(),
            mask_constants.clone(),
            state_type.clone(),
            single_bit,
        )?);
    }

    if single_bit {
        return Ok(mappings);
    }
    let mut mappings_arr = inliner
        .output_graph()
        .create_vector(mappings[0].get_type()?, mappings)?
        .vector_to_array()?;
    let shape_len = mappings_arr.get_type()?.get_dimensions().len();
    let mut permutation: Vec<u64> = (1..shape_len).map(|x| x as u64).collect();
    permutation.rotate_left(2);
    permutation.insert(0, 0);
    mappings_arr = mappings_arr.permute_axes(permutation)?;
    let mut final_mappings = vec![];
    for i in 0..inputs_len {
        final_mappings.push(mappings_arr.get(vec![i])?);
    }
    Ok(final_mappings)
}

#[cfg(test)]
mod tests {
    // Note: we test basic behavior here, and rely on the general inliner tests
    // for the end-to-end behavior testing.
    use super::*;
    use crate::data_values::Value;
    use crate::graphs::create_context;
    use crate::inline::inline_test_utils::{build_test_data, MockInlineState};

    #[test]
    fn test_small_state_iterate_too_many_bits() {
        || -> Result<()> {
            let c = create_context()?;
            let g = c.create_graph()?;
            let initial_state = g.constant(
                array_type(vec![10], BIT),
                Value::from_flattened_array(&vec![0; 10], BIT)?,
            )?;
            let input_vals = vec![1; 5];
            let mut inputs = vec![];
            for i in input_vals {
                let val = g.constant(scalar_type(BIT), Value::from_scalar(i, BIT)?)?;
                inputs.push(val.clone());
            }
            let inputs_node = g.create_vector(scalar_type(BIT), inputs.clone())?;
            let mut inliner = MockInlineState {
                fake_graph: g.clone(),
                inputs: vec![],
                inline_graph_calls: vec![],
                returned_nodes: vec![],
            };
            let g_inline = c.create_graph()?;
            let empty = g_inline.create_tuple(vec![])?;
            g_inline.set_output_node(g_inline.create_tuple(vec![empty.clone(), empty.clone()])?)?;
            let res = inline_iterate_small_state(
                false,
                DepthOptimizationLevel::Extreme,
                g_inline.clone(),
                initial_state.clone(),
                inputs_node.clone(),
                &mut inliner,
            );
            assert!(res.is_err());
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_small_state_iterate_nonempty_output() {
        || -> Result<()> {
            let c = create_context()?;
            let (g, initial_state, inputs_node, _input_vals) = build_test_data(c.clone(), BIT)?;
            let mut inliner = MockInlineState {
                fake_graph: g.clone(),
                inputs: vec![],
                inline_graph_calls: vec![],
                returned_nodes: vec![],
            };
            let g_inline = c.create_graph()?;
            let one_bit = g_inline.input(scalar_type(BIT))?;
            g_inline
                .set_output_node(g_inline.create_tuple(vec![one_bit.clone(), one_bit.clone()])?)?;
            inline_iterate_small_state(
                true,
                DepthOptimizationLevel::Extreme,
                g_inline.clone(),
                initial_state.clone(),
                inputs_node.clone(),
                &mut inliner,
            )?;
            assert_eq!(inliner.inputs.len(), 15);
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_small_state_iterate_valid_case() {
        || -> Result<()> {
            let c = create_context()?;
            let (g, initial_state, inputs_node, _input_vals) = build_test_data(c.clone(), BIT)?;
            let mut inliner = MockInlineState {
                fake_graph: g.clone(),
                inputs: vec![],
                inline_graph_calls: vec![],
                returned_nodes: vec![],
            };
            let g_inline = c.create_graph()?;
            let one_bit = g_inline.input(scalar_type(BIT))?;
            let empty = g_inline.create_tuple(vec![])?;
            g_inline
                .set_output_node(g_inline.create_tuple(vec![one_bit.clone(), empty.clone()])?)?;
            inline_iterate_small_state(
                true,
                DepthOptimizationLevel::Extreme,
                g_inline.clone(),
                initial_state.clone(),
                inputs_node.clone(),
                &mut inliner,
            )?;
            assert_eq!(inliner.inline_graph_calls.len(), 5 * 2);
            Ok(())
        }()
        .unwrap();
    }
}
