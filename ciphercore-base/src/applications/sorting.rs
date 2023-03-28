//! Sorting of an array
use crate::custom_ops::CustomOperation;
use crate::data_types::{array_type, scalar_size_in_bits, ScalarType, Type, BIT};
use crate::errors::Result;
use crate::graphs::*;
use crate::ops::sorting::Sort;

/// Creates a graph that sorts an array of bitstrings of length b using [Batcher's algorithm](https://math.mit.edu/~shor/18.310/batcher.pdf).
///
/// # Arguments
///
/// * `context` - context where a sorting graph should be created
/// * `k` - number of elements of an array (i.e., 2<sup>k</sup>)
/// * `b` - length of input bitstrings
/// * `signed_comparison` - Boolean value indicating whether input bitstrings represent signed or unsigned integers
///
/// # Returns
///
/// Graph that sorts an array of bitstrings
pub fn create_binary_batchers_sorting_graph(
    context: Context,
    k: u32,
    b: u64,
    signed_comparison: bool,
) -> Result<Graph> {
    // Create a graph in a given context that will be used for sorting
    let b_graph = context.create_graph()?;
    // Number of bit strings equal to 2<sup>k</sup>
    let n = 2_u64.pow(k);
    // Create an input node accepting binary arrays of shape [n, b]
    let i_a = b_graph.input(Type::Array(vec![n, b], BIT))?;
    // Sort bitstrings of length b
    let sorted_node = b_graph.custom_op(
        CustomOperation::new(Sort {
            k,
            b,
            signed_comparison,
        }),
        vec![i_a],
    )?;
    // Before computation every graph should be finalized, which means that it should have a designated output node
    // This can be done by calling `g.set_output_node(output)?` or as below
    b_graph.set_output_node(sorted_node)?;
    // Finalization checks that the output node of the graph g is set. After finalization the graph can't be changed
    b_graph.finalize()?;

    Ok(b_graph)
}

/// Creates a graph that sorts an array using [Batcher's algorithm](https://math.mit.edu/~shor/18.310/batcher.pdf).
///
/// # Arguments
///
/// * `context` - context where a sorting graph should be created
/// * `k` - number of elements of an array (i.e., 2<sup>k</sup>)
/// * `st` - scalar type of array elements
///
/// # Returns
///
/// Graph that sorts an array
pub fn create_batchers_sorting_graph(context: Context, k: u32, st: ScalarType) -> Result<Graph> {
    // Create a graph in a given context that will be used for sorting
    let b_graph = context.create_graph()?;
    // To create inputs nodes, compute the bitsize of the input scalar type
    let b = scalar_size_in_bits(st);
    // Boolean value indicating whether input bitstrings represent signed or unsigned integers
    let signed_comparison = st.get_signed();
    // Number of bit strings equal to 2<sup>k</sup>
    let n = 2_u64.pow(k);
    // Define the input node with an array of n integers
    let i = b_graph.input(array_type(vec![n], st))?;
    // If the given scalar type is non-binary, convert input integers to bits.
    let i_a = if st == BIT {
        // Sort custom operation accepts only binary arrays of shape [n, b]
        i.reshape(array_type(vec![n, 1], BIT))?
    } else {
        i.a2b()?
    };
    // Sort bitstrings of length b
    let sorted_node = b_graph.custom_op(
        CustomOperation::new(Sort {
            k,
            b,
            signed_comparison,
        }),
        vec![i_a],
    )?;
    // Convert output from the binary form to the arithmetic form
    let output = if st != BIT {
        sorted_node.b2a(st)?
    } else {
        sorted_node
    };
    // Before computation every graph should be finalized, which means that it should have a designated output node
    // This can be done by calling `g.set_output_node(output)?` or as below
    b_graph.set_output_node(output)?;
    // Finalization checks that the output node of the graph g is set. After finalization the graph can't be changed
    b_graph.finalize()?;

    Ok(b_graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom_ops::run_instantiation_pass;
    use crate::data_types::{ScalarType, BIT, INT64, UINT16, UINT32, UINT64};
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::random::PRNG;
    use std::cmp::Reverse;

    /// Helper function to test the sorting network graph for large inputs
    /// Testing is done by first sorting it with the given graph and then
    /// comparing its result with the non-graph-sorted result
    ///
    /// # Arguments
    ///
    /// * `k` - number of elements of an array (i.e., 2<sup>k</sup>)
    /// * `st` - scalar type of array elements
    fn test_large_vec_batchers_sorting(k: u32, st: ScalarType) -> Result<()> {
        let context = create_context()?;
        let graph: Graph = create_batchers_sorting_graph(context.clone(), k, st)?;
        context.set_main_graph(graph.clone())?;
        context.finalize()?;

        let mapped_c = run_instantiation_pass(graph.get_context())?;

        let seed = b"\xB6\xD7\x1A\x2F\x88\xC1\x12\xBA\x3F\x2E\x17\xAB\xB7\x46\x15\x9A";
        let mut prng = PRNG::new(Some(seed.clone()))?;
        let array_t: Type = array_type(vec![2_u64.pow(k)], st);
        let data = prng.get_random_value(array_t.clone())?;
        if st.get_signed() {
            let data_v_i64 = data.to_flattened_array_i64(array_t.clone())?;
            let result = random_evaluate(mapped_c.mappings.get_graph(graph), vec![data])?
                .to_flattened_array_i64(array_t)?;
            let mut sorted_data = data_v_i64;
            sorted_data.sort_unstable();
            assert_eq!(sorted_data, result);
        } else {
            let data_v_u64 = data.to_flattened_array_u64(array_t.clone())?;
            let result = random_evaluate(mapped_c.mappings.get_graph(graph), vec![data])?
                .to_flattened_array_u64(array_t)?;
            let mut sorted_data = data_v_u64;
            sorted_data.sort_unstable();
            assert_eq!(sorted_data, result);
        }
        Ok(())
    }

    /// Helper function to test the sorting network graph for large inputs
    /// Testing is done by first sorting it with the given graph and then
    /// comparing its result with the non-graph-sorted result
    ///
    /// # Arguments
    ///
    /// * `k` - number of elements of an array (i.e., 2<sup>k</sup>)
    /// * `st` - scalar type of array elements
    fn test_batchers_sorting_graph_helper(k: u32, st: ScalarType, data: Vec<u64>) -> Result<()> {
        let context = create_context()?;
        let graph: Graph = create_batchers_sorting_graph(context.clone(), k, st)?;
        context.set_main_graph(graph.clone())?;
        context.finalize()?;

        let mapped_c = run_instantiation_pass(graph.get_context())?;

        let v_a = Value::from_flattened_array(&data, st)?;
        let result = random_evaluate(mapped_c.mappings.get_graph(graph), vec![v_a])?
            .to_flattened_array_u64(array_type(vec![data.len() as u64], st))?;
        let mut sorted_data = data;
        sorted_data.sort_unstable();
        assert_eq!(sorted_data, result);
        Ok(())
    }

    /// This function tests the well-formed sorting graph for its correctness
    /// Parameters varied are k, st and the input data could be unsorted,
    /// sorted or sorted in a decreasing order.
    #[test]
    fn test_wellformed_batchers_sorting_graph() -> Result<()> {
        let mut data = vec![65535, 0, 2, 32768];
        test_batchers_sorting_graph_helper(2, UINT16, data.clone())?;
        data.sort_unstable();
        test_batchers_sorting_graph_helper(2, UINT16, data.clone())?;
        data.sort_by_key(|w| Reverse(*w));
        test_batchers_sorting_graph_helper(2, UINT16, data.clone())?;

        let data = vec![548890456, 402403639693304868, u64::MAX, 999790788];
        test_batchers_sorting_graph_helper(2, UINT64, data.clone())?;

        let data = vec![643082556];
        test_batchers_sorting_graph_helper(0, UINT32, data.clone())?;

        let data = vec![1, 0, 0, 1];
        test_batchers_sorting_graph_helper(2, BIT, data.clone())?;

        test_large_vec_batchers_sorting(7, BIT)?;
        test_large_vec_batchers_sorting(4, UINT64)?;
        test_large_vec_batchers_sorting(4, INT64)?;

        Ok(())
    }
}
