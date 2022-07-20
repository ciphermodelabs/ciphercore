//! Intersection of two sets represented as arrays
use crate::applications::sorting::create_binary_batchers_sorting_graph;
use crate::custom_ops::CustomOperation;
use crate::custom_ops::Or;
use crate::data_types::{array_type, scalar_size_in_bits, vector_type, ScalarType, Type, BIT};
use crate::errors::Result;
use crate::graphs::*;
use crate::ops::comparisons::Equal;
use crate::ops::multiplexer::Mux;
use crate::ops::utils::zeros_like;
use crate::ops::utils::{pull_out_bits, put_in_bits};

// Depending on the input bit node, selects either an input node or a dummy node with zeros.
// In addition, it attaches the input bit to the selected node.
//
// The first input node is supposed to contain a one-dimensional binary array.
fn select_duplicate_helper(node: Node, bit_node: Node) -> Result<Node> {
    // Extract the graph of the input node
    let g = node.get_graph();
    // Constant node with zero values
    let dummy_node = zeros_like(node.clone())?;
    // Reshape the bit node to correctly select input or dummy elements with bits in the bit node
    // The bit node has shape [n], while the input node and the dummy node have shape [n, b].
    // Due to [the NumPy broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html),
    // the bit node should have shape [n, 1] to select elements row-wise.
    let mut bit_shape = bit_node.get_type()?.get_dimensions();
    bit_shape.push(1);
    let reshaped_bit_node = bit_node.reshape(array_type(bit_shape.clone(), BIT))?;
    // Depending on the equality bit choose the first node or the dummy node
    let selected_node = g.custom_op(
        CustomOperation::new(Mux {}),
        vec![reshaped_bit_node, node.clone(), dummy_node],
    )?;
    // Extract the shape of the input node
    let input_shape = node.get_type()?.get_shape();
    // Extract the number of bits of each bitstring in the input node
    let b = input_shape[input_shape.len() - 1];
    // Put bits of the selected node (input or dummy one) to the first dimension,
    // i.e. change its shape from [..., b] to [b, ...]
    let bits_selected_node = pull_out_bits(selected_node)?.array_to_vector()?;

    // Merge the content of the selected node and the bit_node
    let selected_and_bits = g
        .create_tuple(vec![bits_selected_node, bit_node])?
        .reshape(vector_type(b + 1, array_type(bit_shape, BIT)))?
        .vector_to_array()?;

    // Put bits of the selected node with bits to the last dimension,
    // i.e. change its shape from [b, ...] to [..., b].
    put_in_bits(selected_and_bits)?.array_to_vector()
}

// Checks whether two input nodes contain the same value.
// If yes, the first node with the attached bit 1 is returned; otherwise, returns a node with a dummy value (i.e. zeros) and the zero bit.
fn select_duplicate_2(node1: Node, node2: Node) -> Result<Node> {
    let g = node1.get_graph();
    // Compute the equality bit
    let equality_bit_node =
        g.custom_op(CustomOperation::new(Equal {}), vec![node1.clone(), node2])?;
    // Select the first node or a dummy node with attached bits
    select_duplicate_helper(node1, equality_bit_node)
}

// Checks whether three input nodes contain a pair of adjacent nodes with the same value.
// If yes, the second node with the attached bit 1 is returned; otherwise, returns a node with a dummy value (i.e. zeros) and the zero bit.
fn select_duplicate_3(node1: Node, node2: Node, node3: Node) -> Result<Node> {
    let g = node1.get_graph();
    // Compute the equality bit of the first two nodes
    let equality_bit_1_2 =
        g.custom_op(CustomOperation::new(Equal {}), vec![node1, node2.clone()])?;
    // Compute the equality bit of the last two nodes
    let equality_bit_2_3 =
        g.custom_op(CustomOperation::new(Equal {}), vec![node2.clone(), node3])?;
    // Compute the bit indicating that there is a pair of adjacent nodes with the same value.
    let equal_pair_bit = g.custom_op(
        CustomOperation::new(Or {}),
        vec![equality_bit_1_2, equality_bit_2_3],
    )?;
    // Select the second node or a dummy node with attached bits
    select_duplicate_helper(node2, equal_pair_bit)
}

/// Attaches nodes to an input graph that computes the set intersection of two sets presented as arrays of binary strings.
///
/// # Arguments
///
/// * `g` - non-finalized graph to attach sorting nodes
/// * `sorting_2n_g` - graph that sorts a 2<sup>k+1</sup>-dimensional binary array
/// * `sorting_n_g` - graph that sorts a 2<sup>k</sup>-dimensional binary array
/// * `k` - number of elements in each input array (i.e., 2<sup>k</sup>)
/// * `b` - length of input bitstrings
///
/// # Returns
///
/// Node containing the output of set intersection of two binary arrays
fn attach_binary_set_intersection(
    g: Graph,
    sorting_2n_g: Graph,
    sorting_n_g: Graph,
    k: u32,
    b: u64,
) -> Result<Node> {
    let n = 2_u64.pow(k);
    let input_type = Type::Array(vec![n, b], BIT);
    let i1 = g.input(input_type.clone())?;
    let i2 = g.input(input_type)?;

    // Stack the input arrays into an array of dimension [2*n, b]
    let stacked_input = g
        .stack(vec![i1, i2], vec![2])?
        .reshape(Type::Array(vec![2 * n, b], BIT))?;

    // Sort the resulting big array
    let sorted = g.call(sorting_2n_g, vec![stacked_input])?;

    // Extract duplicates, i.e. elements of the intersection
    let mut duplicates_vec = vec![];
    // Check consecutive triples of the sorted array and extract duplicates.
    // An extracted node is accompanied by a bit indicating whether this node contains an intersection element.
    // Each triple except for the first one contains the last element of the previous one.
    if n > 2 {
        let node1 = sorted.get_slice(vec![SliceElement::SubArray(
            Some(0),
            Some(2 * n as i64 - 2),
            Some(2),
        )])?;
        let node2 = sorted.get_slice(vec![SliceElement::SubArray(
            Some(1),
            Some(2 * n as i64 - 1),
            Some(2),
        )])?;
        let node3 = sorted.get_slice(vec![SliceElement::SubArray(
            Some(2),
            Some(2 * n as i64),
            Some(2),
        )])?;
        let duplicates = select_duplicate_3(node1, node2, node3)?;
        duplicates_vec.push(duplicates);
    }
    // The last array element is not touched by the above code.
    // Check the last pair of the sorted array and extract duplicates.
    let node1 = sorted.get(vec![2 * n - 2])?;
    let node2 = sorted.get(vec![2 * n - 1])?;
    let duplicate = select_duplicate_2(node1, node2)?;
    duplicates_vec.push(duplicate);
    // Create an array of dimension [n, b+1] from extracted duplicates and the related bits
    let binary_duplicates = g
        .create_tuple(duplicates_vec)?
        .reshape(vector_type(n, array_type(vec![b + 1], BIT)))?
        .vector_to_array()?;

    // Sort once more to hide positions of intersection elements
    g.call(sorting_n_g, vec![binary_duplicates])
}

/// Creates a graph that computes the set intersection of two sets presented as arrays of binary strings.
///
/// The algorithm is a variant of [the HEK protocol](https://homes.luddy.indiana.edu/yh33/mypub/psi.pdf).
/// Its steps include:
/// * stacking two input array together,
/// * sorting the resulting array,
/// * extractions of possible duplicates (i.e. elements of the intersection)
/// by checking for equal nodes in consecutive elements of the sorted array,
/// * sorting the resulting array to hide positions of intersection elements.
///
/// Input arrays are binary and contain 2<sup>k</sup> bitstrings.
/// Each input array should represent a set, i.e. it should not contain duplicates.
///
/// The output of this graph is a tuple of two arrays of size 2<sup>k</sup>:
/// * the first array is binary and contains several (possibly zero) leading zeros followed by binary elements of the intersection (sorted in the ascending lexicographic order);
/// * the second array is binary; it's i-th element is 1 if and only if the i-th element of the first array belongs to the set intersection.
///
/// # Arguments
///
/// * `context` - context where a set intersection graph should be created
/// * `k` - number of elements in each input array (i.e., 2<sup>k</sup>)
/// * `b` - length of input bitstrings
///
/// # Returns
///
/// Graph that intersects sets
pub fn create_binary_set_intersection_graph(context: Context, k: u32, b: u64) -> Result<Graph> {
    let sorting_2n_g = create_binary_batchers_sorting_graph(context.clone(), k + 1, b, false)?;
    let sorting_n_g = create_binary_batchers_sorting_graph(context.clone(), k, b + 1, false)?;

    let g = context.create_graph()?;

    // Attach nodes that perform set intersection
    let binary_duplicates_sorted =
        attach_binary_set_intersection(g.clone(), sorting_2n_g, sorting_n_g, k, b)?;

    // Convert elements to the corresponding scalar types and
    // extract bits indicating whether the related element is in the intersection
    let binary_intersection = binary_duplicates_sorted.get_slice(vec![
        SliceElement::Ellipsis,
        SliceElement::SubArray(None, Some(b as i64), None),
    ])?;
    let intersection_bits = binary_duplicates_sorted.get_slice(vec![
        SliceElement::Ellipsis,
        SliceElement::SingleIndex(b as i64),
    ])?;

    let output = g.create_tuple(vec![binary_intersection, intersection_bits])?;
    output.set_as_output()?;
    g.finalize()?;

    Ok(g)
}

/// Creates a graph that computes the set intersection of two sets presented as arrays.
///
/// The algorithm is a variant of [the HEK protocol](https://homes.luddy.indiana.edu/yh33/mypub/psi.pdf).
/// Its steps include:
/// * stacking two input array together,
/// * sorting the resulting array,
/// * extractions of possible duplicates (i.e. elements of the intersection)
/// by checking for equal nodes in consecutive elements of the sorted array,
/// * sorting the resulting array to hide positions of intersection elements.
///
/// Input arrays are binary and contain 2<sup>k</sup> bitstrings.
/// Each input array should represent a set, i.e. it should not contain duplicates.
///
/// The output of this graph is a tuple of two arrays of size 2<sup>k</sup>:
/// * the first array contains several (possibly zero) leading zeros followed by positive and then negative elements of the intersection (both sorted in the ascending order);
/// * the second array is binary; it's i-th element is 1 if and only if the i-th element of the first array belongs to the set intersection.
///
/// # Arguments
///
/// * `context` - context where a set intersection graph should be created
/// * `k` - number of elements in each input array (i.e., 2<sup>k</sup>)
/// * `st` - scalar type of array elements
///
/// # Returns
///
/// Graph that intersects sets
pub fn create_set_intersection_graph(context: Context, k: u32, st: ScalarType) -> Result<Graph> {
    let b = scalar_size_in_bits(st.clone());
    let sorting_2n_g =
        create_binary_batchers_sorting_graph(context.clone(), k + 1, b, st.get_signed())?;
    let sorting_n_g = create_binary_batchers_sorting_graph(context.clone(), k, b + 1, false)?;

    let g = context.create_graph()?;

    // Attach nodes that perform set intersection
    let binary_duplicates_sorted =
        attach_binary_set_intersection(g.clone(), sorting_2n_g, sorting_n_g, k, b)?;

    // Convert elements to the corresponding scalar types and
    // extract bits indicating whether the related element is in the intersection
    let binary_intersection = binary_duplicates_sorted.get_slice(vec![
        SliceElement::Ellipsis,
        SliceElement::SubArray(None, Some(b as i64), None),
    ])?;
    let intersection_bits = binary_duplicates_sorted.get_slice(vec![
        SliceElement::Ellipsis,
        SliceElement::SingleIndex(b as i64),
    ])?;
    let intersection = if st != BIT {
        binary_intersection.b2a(st)?
    } else {
        binary_intersection
    };

    let output = g.create_tuple(vec![intersection, intersection_bits])?;
    output.set_as_output()?;
    g.finalize()?;

    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom_ops::run_instantiation_pass;
    use crate::data_types::{array_type, ScalarType, INT64, UINT64};
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use std::ops::Not;

    fn intersection_helper<T: TryInto<u64> + Not<Output = T> + TryInto<u8> + Copy>(
        k: u32,
        st: ScalarType,
        data1: Vec<T>,
        data2: Vec<T>,
        expected_values: Vec<u64>,
        expected_bits: Vec<u64>,
    ) -> Result<()> {
        let context = create_context()?;
        let graph: Graph = create_set_intersection_graph(context.clone(), k, st.clone())?;
        context.set_main_graph(graph.clone())?;
        context.finalize()?;

        let mapped_c = run_instantiation_pass(graph.get_context())?;

        let v_1 = Value::from_flattened_array(&data1, st.clone())?;
        let v_2 = Value::from_flattened_array(&data2, st.clone())?;
        let result_tuple =
            random_evaluate(mapped_c.mappings.get_graph(graph), vec![v_1, v_2])?.to_vector()?;
        let result_values =
            result_tuple[0].to_flattened_array_u64(array_type(vec![data1.len() as u64], st))?;
        let result_bits =
            result_tuple[1].to_flattened_array_u64(array_type(vec![data1.len() as u64], BIT))?;
        assert_eq!(expected_values, result_values);
        assert_eq!(expected_bits, result_bits);
        Ok(())
    }

    #[test]
    fn test_set_intersection() {
        intersection_helper(0, UINT64, vec![0], vec![9], vec![0], vec![0]).unwrap();
        intersection_helper(0, UINT64, vec![9], vec![9], vec![9], vec![1]).unwrap();
        intersection_helper(
            2,
            UINT64,
            vec![0, 2, 3, 4],
            vec![9, 3, 6, 5],
            vec![0, 0, 0, 3],
            vec![0, 0, 0, 1],
        )
        .unwrap();
        intersection_helper(
            2,
            UINT64,
            vec![0, 2, 3, 4],
            vec![4, 3, 2, 0],
            vec![0, 2, 3, 4],
            vec![1, 1, 1, 1],
        )
        .unwrap();
        intersection_helper(
            2,
            UINT64,
            vec![3, 6, 2, 5],
            vec![4, 1, 8, 9],
            vec![0, 0, 0, 0],
            vec![0, 0, 0, 0],
        )
        .unwrap();
        intersection_helper(
            2,
            UINT64,
            vec![u64::MAX, 3, 4, u64::MAX - 1],
            vec![4, u64::MAX, 8, 9],
            vec![0, 0, 4, u64::MAX],
            vec![0, 0, 1, 1],
        )
        .unwrap();

        intersection_helper(0, INT64, vec![0], vec![-5], vec![0], vec![0]).unwrap();
        intersection_helper(0, INT64, vec![-5], vec![-5], vec![u64::MAX - 4], vec![1]).unwrap();
        intersection_helper(
            2,
            INT64,
            vec![0, 2, -3, -4],
            vec![0, 2, -3, -5],
            vec![0, 0, 2, u64::MAX - 2],
            vec![0, 1, 1, 1],
        )
        .unwrap();
        intersection_helper(
            2,
            INT64,
            vec![0, 2, -3, -4],
            vec![0, 2, -3, -4],
            vec![0, 2, u64::MAX - 3, u64::MAX - 2],
            vec![1, 1, 1, 1],
        )
        .unwrap();
        intersection_helper(
            2,
            INT64,
            vec![1, 2, -3, -4],
            vec![5, 6, -7, -8],
            vec![0, 0, 0, 0],
            vec![0, 0, 0, 0],
        )
        .unwrap();
        intersection_helper(
            2,
            INT64,
            vec![1, 2, -3, -4],
            vec![5, 6, -3, -8],
            vec![0, 0, 0, u64::MAX - 2],
            vec![0, 0, 0, 1],
        )
        .unwrap();

        //malformed sets
        intersection_helper(
            2,
            UINT64,
            vec![1, 1, 3, 4],
            vec![5, 6, 3, 8],
            vec![0, 0, 1, 3], // the expected result isn't the intersection
            vec![0, 0, 1, 1],
        )
        .unwrap();
        intersection_helper(
            2,
            UINT64,
            vec![1, 2, 3, 8],
            vec![5, 6, 8, 8],
            vec![0, 0, 8, 8], // the expected result isn't the intersection
            vec![0, 0, 1, 1],
        )
        .unwrap();
    }
}
