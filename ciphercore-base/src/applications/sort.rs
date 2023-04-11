//! Sorting of an array
use crate::custom_ops::CustomOperation;
use crate::data_types::{array_type, ScalarType, BIT};
use crate::errors::Result;
use crate::graphs::*;
use crate::ops::integer_key_sort::SortByIntegerKey;

/// Creates a graph that sorts an array using [Radix Sort MPC protocol](https://eprint.iacr.org/2019/695.pdf).
///
/// # Arguments
///
/// * `context` - context where a sorting graph should be created
/// * `n` - number of elements of an array
/// * `st` - scalar type of array elements
///
/// # Returns
///
/// Graph that sorts an array
pub fn create_sort_graph(context: Context, n: u64, st: ScalarType) -> Result<Graph> {
    // Create a graph in a given context that will be used for sorting
    let graph = context.create_graph()?;
    // Define the input node with an array of n integers
    let input = graph.input(array_type(vec![n], st))?;
    // Create named a tuple as required by the interface.
    let key = "key".to_string();
    let node = graph.create_named_tuple(vec![(key.clone(), input)])?;
    // Sort an array
    let sorted_node = graph.custom_op(
        CustomOperation::new(SortByIntegerKey { key: key.clone() }),
        vec![node],
    )?;
    // Extract result from tuple by key.
    let output = sorted_node.named_tuple_get(key)?;

    // Before computation every graph should be finalized, which means that it should have a designated output node
    // This can be done by calling `g.set_output_node(output)?` or as below
    graph.set_output_node(output)?;
    // Finalization checks that the output node of the graph g is set. After finalization the graph can't be changed
    graph.finalize()?;

    Ok(graph)
}

/// Creates a graph that sorts an array of bitstrings using [Radix Sort MPC protocol](https://eprint.iacr.org/2019/695.pdf).
///
/// # Arguments
///
/// * `context` - context where a sorting graph should be created
/// * `n` - number of elements of an array
/// * `b` - length of bitstrings
///
/// # Returns
///
/// Graph that sorts an array
pub fn create_binary_sort_graph(context: Context, n: u64, b: u64) -> Result<Graph> {
    // Create a graph in a given context that will be used for sorting
    let graph = context.create_graph()?;
    // Define the input node with an array of n integers
    let input = graph.input(array_type(vec![n, b], BIT))?;
    // Create a named tuple as required by the interface.
    let key = "key".to_string();
    let node = graph.create_named_tuple(vec![(key.clone(), input)])?;
    // Sort an array
    let sorted_node = node.sort(key.clone())?;
    // Extract the result from the tuple by key.
    let output = sorted_node.named_tuple_get(key)?;

    // Before computation every graph should be finalized, which means that it should have a designated output node
    // This can be done by calling `g.set_output_node(output)?` or as below
    graph.set_output_node(output)?;
    // Finalization checks that the output node of the graph g is set. After finalization the graph can't be changed
    graph.finalize()?;

    Ok(graph)
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
    /// * `n` - number of elements of an array
    /// * `st` - scalar type of array elements
    fn test_large_vec_sort(n: u64, st: ScalarType) -> Result<()> {
        let context = create_context()?;
        let graph: Graph = create_sort_graph(context.clone(), n, st)?;
        context.set_main_graph(graph.clone())?;
        context.finalize()?;

        let mapped_c = run_instantiation_pass(graph.get_context())?;

        let seed = b"\xB6\xD7\x1A\x2F\x88\xC1\x12\xBA\x3F\x2E\x17\xAB\xB7\x46\x15\x9A";
        let mut prng = PRNG::new(Some(seed.clone()))?;
        let array_t = array_type(vec![n], st);
        let data = prng.get_random_value(array_t.clone())?;
        if st.is_signed() {
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
    /// * `n` - number of elements of an array
    /// * `st` - scalar type of array elements
    fn test_sort_graph_helper(n: u64, st: ScalarType, data: Vec<u64>) -> Result<()> {
        let context = create_context()?;
        let graph: Graph = create_sort_graph(context.clone(), n, st)?;
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
    fn test_sort_graph() -> Result<()> {
        let mut data = vec![65535, 0, 2, 32768];
        test_sort_graph_helper(4, UINT16, data.clone())?;
        data.sort_unstable();
        test_sort_graph_helper(4, UINT16, data.clone())?;
        data.sort_by_key(|w| Reverse(*w));
        test_sort_graph_helper(4, UINT16, data.clone())?;

        let data = vec![548890456, 402403639693304868, u64::MAX, 999790788];
        test_sort_graph_helper(4, UINT64, data.clone())?;

        let data = vec![643082556];
        test_sort_graph_helper(1, UINT32, data.clone())?;

        let data = vec![1, 0, 0, 1];
        test_sort_graph_helper(4, BIT, data.clone())?;

        test_large_vec_sort(1000, BIT)?;
        test_large_vec_sort(1000, UINT64)?;
        test_large_vec_sort(1000, INT64)?;

        Ok(())
    }
}
