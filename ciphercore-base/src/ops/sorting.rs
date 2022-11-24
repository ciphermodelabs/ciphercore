//! Sorting of an array
use crate::custom_ops::{CustomOperation, CustomOperationBody};
use crate::data_types::{array_type, vector_type, Type, BIT};
use crate::errors::Result;
use crate::graphs::SliceElement::SubArray;
use crate::graphs::*;
use crate::ops::min_max::{Max, Min};

use serde::{Deserialize, Serialize};

/// A structure that defines the custom operation Sort that implements sorting of binary strings representing signed or unsigned numbers.
///
/// This operation sorts an array of `k` bitstrings with `b` bits in  using [Batcher's algorithm](https://math.mit.edu/~shor/18.310/batcher.pdf).
/// Bitstrings can be sorted in ascending lexicographic order, or they can be interpreted and sorted in ascending order as signed integers,
/// which can be controlled by the `signed_comparison` parameter.  
///
/// To use this and other custom operations in computation graphs, see [Graph::custom_op].
///
/// # Custom operation arguments
///
/// - Node containing an array of binary strings to sort
///
/// # Custom operation returns
///
/// New Sort node
///
/// # Example
///
/// ```
/// # use ciphercore_base::graphs::create_context;
/// # use ciphercore_base::data_types::{array_type, BIT};
/// # use ciphercore_base::custom_ops::{CustomOperation};
/// # use ciphercore_base::ops::sorting::Sort;
/// let c = create_context().unwrap();
/// let g = c.create_graph().unwrap();
/// let t = array_type(vec![4, 8], BIT);
/// let n1 = g.input(t.clone()).unwrap();
/// let n2 = g.custom_op(CustomOperation::new(Sort {k: 2, b: 8, signed_comparison: true}), vec![n1]).unwrap();
/// ```
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct Sort {
    /// number of elements of an array (i.e., 2<sup>k</sup>)
    pub k: u32,
    /// length of input bitstrings
    pub b: u64,
    /// Boolean value indicating whether input bitstrings represent signed or unsigned integers
    pub signed_comparison: bool,
}

#[typetag::serde]
impl CustomOperationBody for Sort {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        if arguments_types.len() != 1 {
            return Err(runtime_error!("Sort accepts only 1 argument"));
        }
        if arguments_types[0].get_scalar_type() != BIT {
            return Err(runtime_error!(
                "Sort accepts only arrays of bitstrings as input"
            ));
        }
        let n = 2_u64.pow(self.k);
        if arguments_types[0] != array_type(vec![n, self.b], BIT) {
            return Err(runtime_error!(
                "Sort accepts only arrays of {} {}-bit bitstrings as input",
                n,
                self.b
            ));
        }

        let g = context.create_graph()?;
        // NOTE: The implementation is based on the 'bottom up' approach as described in
        // https://math.mit.edu/~shor/18.310/batcher.pdf.
        // Commenting about the initial few shape changes done with the help of a
        // 16 element array example
        // Create an input node accepting binary arrays of shape [n, b]
        let i_a = g.input(Type::Array(vec![n, self.b], BIT))?;
        // Stash of nodes uses as input of each iteration of the following loop
        let mut stage_ops = vec![i_a];
        // The following loop, over 'it', corresponds to sorting (SORT()) operation
        // in https://math.mit.edu/~shor/18.310/batcher.pdf.
        for it in 1..(self.k + 1) {
            let num_classes: u64 = 2_u64.pow(it);
            let num_class_reps = n / num_classes;
            let data_to_sort = stage_ops[(it - 1) as usize].clone();
            // For it==1, we are sorting into pairs i.e. we will have pairs of sorted keys
            // For it==2, we are creating sorted groups of size 4
            // For it==3, we are creating sorted groups of size 8
            // For it==4, we are creating sorted groups of size 16 and so on

            // For the purposes of the discussion, we will temporarily disregard the
            // final dimension i.e. the bit dimension so as to understand how the
            // jiggling of array shape is happening for the elements involved

            // Divide the keys into 2^{it} classes or groups
            let global_a_reshape = g.reshape(
                data_to_sort.clone(),
                array_type(vec![num_class_reps, num_classes, self.b], BIT),
            )?;

            // 1-D Array Indices:                   0  1  2  3      14 15
            // At it==1, we would have 2^1 classes: 0, 1, 0, 1, ..., 0, 1
            // Now, global_a_reshape shape (2-D shape), in terms of indices, looks like:
            // class0|  class1|
            // ______|________|
            //      0|       1|
            //      2|       3|
            //      .|       .|
            //      .|       .|
            //     12|      13|
            //     14|      15|

            // 1-D Array Indices:                   0  1  2  3  4  5      10 11 12 13 14 15
            // At it==2, we would have 2^2 classes: 0, 1, 2, 3, 0, 1, ..., 2, 3, 0, 1, 2, 3
            // Now, 2-D global_a_reshape shape, in terms of indices, looks like:
            // class0  class1  class2  class3
            //      0       1       2       3
            //      4       5       6       7
            //      8       9      10      11
            //      12     13      14      15

            // 1-D Array Indices:     0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
            // At it==3, 2^3 classes: 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7
            // Now, 2-D global_a_reshape shape, in terms of indices, looks like:
            // class0  class1  class2  class3 class4  class5  class6  class7
            //      0       1       2       3      4       5       6       7
            //      8       9      10      11     12      13      14      15

            // Permute the axes to perform the transpose operation
            // This is done so that each row now corresponds to a single class or group
            let mut global_chunks_a = g.permute_axes(global_a_reshape, vec![1, 0, 2])?;
            // Based on 'it' the global_chunks_a shape looks like
            // For it == 1, locations of flat (1-D) indices
            // class0:  0, 2, ..., 12, 14
            // class1:  1, 3, ..., 13, 15

            // For it == 2, locations of flat (1-D) indices
            // class0:  [[0, 4, 8, 12]
            // class1:   [1, 5, 9, 13]
            // class2:   [2, 6, 10, 14]
            // class3:   [3, 7, 11, 15]]

            // For it == 3, locations of flat (1-D) indices
            // class0:  [[0, 8]
            // class1:   [1, 9]
            // class2:   [2, 10]
            // class3:   [3, 11]
            // class4:   [4, 12]
            // class5:   [5, 13]
            // class6:   [6, 14]
            // class7:   [7, 15]]

            let mut intermediate_chunks_a: Vec<Node> = vec![];
            // The below loop, over 'i', corresponds to the MERGE() operation in https://math.mit.edu/~shor/18.310/batcher.pdf
            // - In the 'bottom up' approach, the operations contained in loop are
            // also referred as 'round(s) of comparisons' in https://math.mit.edu/~shor/18.310/batcher.pdf
            // - For groups or classes of size 2^{it}, you would require 'it' rounds
            // of comparisons
            // - The operations are vectorized to leverage the inherent parallelism
            // - For each group or class to be sorted, initially pairs are formed for
            // sorting then groups of 4 are formed for sorting, likewise for 8, 16 and so on.
            // - Technically, here, the number of dimensions are 4, however, we will
            // ignore the innermost dimension that corresponds to bits as it would
            // be handled by the custom_operations Min{} and Max{} and is not as relevant
            // to the Batcher's algorithm logic
            // - Formation of sub-groups of key sizes 2 or 4, 8, 16, ... for each group
            // of size 2^{it} happens along the outermost axis, whose size is
            // referenced here by 'chunks_a_sz_z'
            for i in (0..it).rev() {
                let chunks_a_sz_y = 2_u64.pow(i);
                let chunks_a_sz_z = 2_u64.pow(it - i); //n / (chunks_a_sz_y * num_class_reps);

                // Reshape to create an additional dimension that corresponds to
                // each sub-group (of sizes 2, 4, 8, 16, ..., 2^{it-i}) within
                // original group of 2^{it} keys, which is to be sorted
                let chunks_a = g.reshape(
                    global_chunks_a.clone(),
                    array_type(
                        vec![chunks_a_sz_z, chunks_a_sz_y, num_class_reps, self.b],
                        BIT,
                    ),
                )?;

                // For it==1 and i==0,
                // the two sub-groups are placed side-by-side along for sorting pairs
                // the outermost (Z) axis, Y-axis (height) is 1 and X-axis (breadth) is 8
                // i.e. Z_0 corresponds to class0, Z_1 corresponds to class1 and so on

                // For it==2 and i==1,
                // sorting the groups of 4 by first sorting the pairs within them
                // chunks_a:
                // [ [[0, 4, 8, 12],    Values    [ [[min(0, 1), min(4, 5), min(8, 9), min(12, 13)],
                //    [1, 5, 9, 13]],   =====>       [max(0, 1), max(4, 5), max(8, 9)], max(12, 13)],
                //   [[2, 6, 10, 14],               [[min(2, 3), min(6, 7), min(10, 11), min(14, 15)],
                //    [3, 7, 11, 15]]                [max(2, 3), max(2, 7), max(10, 11), max(14, 15)]]
                // ]                              ]
                //
                // For it==2 and i==0,
                // sorting the groups of 4 by sorting all 4 elements,
                // i.e., Z_0 corresponds to class0, Z_1 corresponds to class1 and so on
                // chunks_a:
                // [
                //      [[min(0, 2), max(0, 2), min(4, 6), max(4, 6)]],
                //      [[min(8, 10), max(8, 10), min(12, 14), max(12, 14)]],
                //      [[min(1, 3), max(1, 3), min(5, 7), max(5, 7)]],
                //      [[min(9, 11), max(9, 11), min(13, 15), max(13, 15)]]
                //
                // ]

                let (chunks_a_shape, chunks_a_scalar_t) = match chunks_a.get_type()? {
                    Type::Array(shape, scalar_type) => (shape, scalar_type),
                    _ => return Err(runtime_error!("Array Type not found")),
                };

                // Get the first class elements i.e. Z_0
                let single_first_element = g.get(chunks_a.clone(), vec![0])?;
                // For it==1, i==0, single_first_element shape would be [1, 8, x]
                // For it==2, i==1, single_first_element shape would be [2, 4, x]
                // For it==2, i==0, single_first_element shape would be [1, 4, x]

                // If first step, then arrange odd-even adjacent pairs of keys into ordered pairs
                if i == it - 1 {
                    // Code to sort the only two chunks/halfs only

                    // Here, we are dealing with just two classes: Z_0 (odds) and Z_1 (evens)
                    // For it==1, i==0,
                    // Z_0 and Z_1 shapes are [1, 8, x]

                    // For it==2, i==1,
                    // Z_0 and Z_1 shapes are [2, 4, x]

                    // Get the group of odd indexed keys from each group or class, i.e., Z_{0}
                    let uu = single_first_element;

                    // Get the group of even indexed keys from each group or class, i.e., Z_{1}
                    let vv = g.get(chunks_a.clone(), vec![1])?;

                    // Get minimums from both the classes
                    let chunks_a_0 = g.custom_op(
                        CustomOperation::new(Min {
                            signed_comparison: self.signed_comparison,
                        }),
                        vec![uu.clone(), vv.clone()],
                    )?;
                    // For it==1, i==0, chunks_a_0 = [[min(0, 1), min(2, 3), ..., min(12, 13), min(14, 15)]]
                    // For it==2, i==1, chunks_a_0 = [[min(0, 2), min(4, 6), min(8, 10), min(12, 14)],
                    //                                [min(1, 3), min(5, 7), min(9, 11), min(13, 15)]]

                    // Get maximums from both the classes
                    let chunks_a_1 = g.custom_op(
                        CustomOperation::new(Max {
                            signed_comparison: self.signed_comparison,
                        }),
                        vec![uu.clone(), vv.clone()],
                    )?;
                    // For it==1, i==0, chunks_a_1 = [[max(0, 1), max(2, 3), ..., max(12, 13), max(14, 15)]]
                    // For it==2, i==1, chunks_a_0 = [[max(0, 2), max(4, 6), max(8, 10), max(12, 14)],
                    //                                [max(1, 3), max(5, 7), max(9, 11), max(13, 15)]]

                    // Collect these maximums and minimums together for reshaping later
                    let a_combined = g.create_tuple(vec![chunks_a_0, chunks_a_1])?;
                    // For it==1, i==0,
                    // a_combined = [(min(0, 1), max(0, 1)), (min(2, 3), max(2, 3)), ..., (min(12, 13), max(12, 13)), (min(14, 15), max(14, 15))]
                    // For it==2, i==1,
                    // a_combined = [[(min(0, 2), max(0, 2)), (min(4, 6), max(4, 6)), (min(8, 10), max(8, 10)), (min(12, 14), max(12, 14))],
                    //               [(min(1, 3), max(1, 3)), (min(5, 7), max(5, 7)), (min(9, 11), max(9, 11)), (min(13, 15), max(13, 15))]]

                    // Reshape these combined elements back into a vector shape
                    let interm_chunks_a = g.reshape(
                        a_combined,
                        vector_type(
                            chunks_a_sz_z,
                            array_type(
                                vec![chunks_a_sz_y, num_class_reps, self.b],
                                chunks_a_scalar_t,
                            ),
                        ),
                    )?;
                    // For it==1, i==0,
                    // i.e., chunks_a's shape [2, 1, 8, x] for further processing
                    // interm_chunks_a =    <[min(0, 1), max(0, 1), min(2, 3), max(2, 3), min(4, 5), max(4, 5), min(6, 7), max(6, 7)]>,
                    //                      <[min(8, 9), max(8, 9), min(10, 11), max(10, 11), min(12, 13), max(12, 13), min(14, 15), max(14, 15)]>

                    // For it==2, i==1,
                    // chunks_a's shape [2, 2, 4, x]
                    // interm_chunks_a = <[ [min(0, 2), max(0, 2), min(4, 6), max(4, 6)],
                    //                      [min(8, 10), max(8, 10), min(12, 14), max(12, 14)] ],
                    //                    [ [min(1, 3), max(1, 3), min(5, 7), max(5, 7)],
                    //                      [min(9, 11), max(9, 11), min(13, 15), max(13, 15)] ]>

                    // Convert these combined elements back to an array of original shape
                    intermediate_chunks_a.push(g.vector_to_array(interm_chunks_a)?);
                    // For it==1, i==0,
                    // i.e., into the chunks_a's shape [2, 1, 8, x] for further processing
                    // intermediate_chunks_a[intermediate_chunks_a.len()-1] =
                    // [
                    //  [[min(0, 1), max(0, 1), min(2, 3), max(2, 3), min(4, 5), max(4, 5), min(6, 7), max(2, 7)]],
                    //
                    //  [[min(8, 9), max(8, 9), min(10, 11), max(10, 11), min(12, 13), max(12, 13), min(14, 15), max(14, 15)]]
                    // ]

                    // For it==2, i==1,
                    // i.e., into the chunks_a's shape [2, 2, 4, x] for further processing
                    // intermediate_chunks_a[intermediate_chunks_a.len()-1] =
                    //  [
                    //      [[min(0, 2), max(0, 2), min(4, 6), max(4, 6)],
                    //       [min(8, 10), max(8, 10), min(12, 14), max(12, 14)]],
                    //      [[min(1, 3), max(1, 3), min(5, 7), max(5, 7)],
                    //       [min(9, 11), max(9, 11), min(13, 15), max(13, 15)]]
                    //  ]
                } else {
                    // This else block corresponds to the COMP() operations
                    // specified within the MERGE() function in (https://math.mit.edu/~shor/18.310/batcher.pdf, p. 3) and
                    // if x_{1}, x_{2}, ..., x_{n} are the keys to be sorted then
                    // this COMP is operated as COMP(x 2 , x 3 ), COMP(x 4 , x 5 ), · · ·
                    // COMP(x n−2 , x n−1 ).
                    // In this case, we would not be considering terminal sub-groups
                    // i.e. Z_{0} and Z_{2^{it-i}-1}

                    // Set the shape of Z_0
                    let a_single_first_elem = g.reshape(
                        single_first_element,
                        array_type(
                            chunks_a_shape[1..chunks_a_shape.len()].to_vec(),
                            chunks_a_scalar_t.clone(),
                        ),
                    )?;
                    // For it==2, i==0,
                    // a_single_first_elem =
                    // [min(0, 2), max(0, 2), min(4, 6), max(4, 6)]

                    // Obtain all the odd components of Z, except the first and last one,
                    // i.e., Z_{i} s.t. 1 <=i < 2^{it-i}-1 && i % 2 == 1
                    let uu =
                        g.get_slice(chunks_a.clone(), vec![SubArray(Some(1), Some(-1), Some(2))])?;
                    // For it==2, i==0, uu shape = [1, 1, 4, x], uu =
                    // [
                    //      [[min(8, 10), max(8, 10), min(12, 14), max(12, 14)]],
                    // ]

                    // Obtain all the even components of Z, except the first one i.e.
                    // Z_{i} s.t. 2 <= i < 2^{it-i} && i % 2 == 0
                    let vv =
                        g.get_slice(chunks_a.clone(), vec![SubArray(Some(2), None, Some(2))])?;
                    // For it==2, i==0, vv shape = [1, 1, 4, x], vv =
                    // [
                    //      [[min(1, 3), max(1, 3), min(5, 7), max(5, 7)]],
                    // ]

                    // Obtain the minimum of these two arrays - uu and vv
                    let chunks_a_evens = g.custom_op(
                        CustomOperation::new(Min {
                            signed_comparison: self.signed_comparison,
                        }),
                        vec![uu.clone(), vv.clone()],
                    )?;
                    // For it==2, i==0, chunks_a_evens shape = [1, 1, 4, x], chunks_a_evens =
                    // [
                    //      [[min(8, 10, 1, 3), min(max(8, 10), max(1, 3)), min(12, 14, 5, 7), min(max(12, 14), max(5, 7))]]
                    // ]

                    // Obtain the maximum of these two arrays - uu and vv
                    let chunks_a_odds = g.custom_op(
                        CustomOperation::new(Max {
                            signed_comparison: self.signed_comparison,
                        }),
                        vec![uu.clone(), vv.clone()],
                    )?;
                    // For it==2, i==0, chunks_a_odds shape = [1, 1, 4, x], chunks_a_odds =
                    // [
                    //      [[max(min(8, 10), min(1, 3)), max(8, 10, 1, 3), max(min(12, 14), min(5, 7)), max(12, 14, 5, 7)]]
                    // ]

                    // Convert the array to vector and remove the extra Z-dimension
                    let v_non_terminal_evens = g.array_to_vector(chunks_a_evens)?;
                    // For it==2, i==0, v_non_terminal_evens shape = [1, 4, x]<1>
                    // v_non_terminal_evens =
                    // <[min(8, 10, 1, 3), min(max(8, 10), max(1, 3)), min(12, 14, 5, 7), min(max(12, 14), max(5, 7))]>

                    // Convert the array to vector and remove the extra Z-dimension
                    let v_non_terminal_odds = g.array_to_vector(chunks_a_odds)?;
                    // For it==2, i==0, v_non_terminal_odds shape = [1, 4, x]<1>
                    // v_non_terminal_odds =
                    // <[max(min(8, 10), min(1, 3)), max(8, 10, 1, 3), max(min(12, 14), min(5, 7)), max(12, 14, 5, 7)]>

                    // Zip both the results together
                    let v_non_term_elems =
                        g.zip(vec![v_non_terminal_evens, v_non_terminal_odds])?;
                    // For it==2, i==0, v_non_term_elems shape = ((1, 4, x)(1, 4, x))<1>
                    // v_non_term_elems =
                    // <(min(8, 10, 1, 3), max(min(8, 10), min(1, 3))),
                    //  (min(max(8, 10), max(1, 3)), max(8, 10, 1, 3)),
                    //  (min(12, 14, 5, 7), max(min(12, 14), min(5, 7))),
                    //  (min(max(12, 14), max(5, 7)), max(12, 14, 5, 7))>

                    // In a similar way to the first element i.e. Z_{0}, extract the last element
                    let single_last_elem = g.get(chunks_a.clone(), vec![chunks_a_shape[0] - 1])?;
                    // For it==2, i==0, single_last_element shape would be [1, 4, x]

                    // Set the shape of Z_{2^{it-i}-1} to [1, 4, x]
                    let a_single_last_elem = g.reshape(
                        single_last_elem,
                        array_type(
                            chunks_a_shape[1..chunks_a_shape.len()].to_vec(),
                            chunks_a_scalar_t.clone(),
                        ),
                    )?;
                    // For it==2, i==0,
                    // a_single_last_elem =
                    //  [min(9, 11), max(9, 11), min(13, 15), max(13, 15)]

                    // Create a tuple of Z: (first element-Z_{0}, vector, last element-Z_{2^{it-i}-1})
                    let v_combined = g.create_tuple(vec![
                        a_single_first_elem,
                        v_non_term_elems,
                        a_single_last_elem,
                    ])?;
                    // For it==2, i==0,
                    // v_combined =
                    // ([min(0, 2), max(0, 2), min(4, 6), max(4, 6)],
                    //  <(min(8, 10, 1, 3), max(min(8, 10), min(1, 3))),
                    //  (min(max(8, 10), max(1, 3)), max(8, 10, 1, 3)),
                    //  (min(12, 14, 5, 7), max(min(12, 14), min(5, 7))),
                    //  (min(max(12, 14), max(5, 7)), max(12, 14, 5, 7))>,
                    //  [min(9, 11), max(9, 11), min(13, 15), max(13, 15)]
                    // )

                    // Reshape the tuple back into vector form
                    let v_chunk_a = g.reshape(
                        v_combined,
                        vector_type(
                            chunks_a_shape[0],
                            array_type(
                                chunks_a_shape[1..chunks_a_shape.len()].to_vec(),
                                chunks_a_scalar_t,
                            ),
                        ),
                    )?;
                    // For it==2, i==0,
                    // v_chunk_a's shape is {[1, 4, x]}<4> i.e. 4 components, each an
                    // array of size [1, 4, x]
                    // v_chunk_a =
                    // <
                    //   [min(0, 2), max(0, 2), min(4, 6), max(4, 6)],
                    //   [min(8, 10, 1, 3), max(min(8, 10), min(1, 3)), min(max(8, 10), max(1, 3)), max(8, 10, 1, 3)],
                    //   [min(12, 14, 5, 7), max(min(12, 14), min(5, 7)), min(max(12, 14), max(5, 7)), max(12, 14, 5, 7),
                    //   [min(9, 11), max(9, 11), min(13, 15), max(13, 15)]
                    // >
                    //

                    // Convert the vector form to the array form
                    intermediate_chunks_a.push(g.vector_to_array(v_chunk_a)?);
                    // For it==2, i==0,
                    // intermediate_chunks_a[intermediate_chunks_a.len()-1] =
                    // [
                    //   [min(0, 2), max(0, 2), min(4, 6), max(4, 6)],
                    //   [min(8, 10, 1, 3), max(min(8, 10), min(1, 3)), min(max(8, 10), max(1, 3)), max(8, 10, 1, 3)],
                    //   [min(12, 14, 5, 7), max(min(12, 14), min(5, 7)), min(max(12, 14), max(5, 7)), max(12, 14, 5, 7),
                    //   [min(9, 11), max(9, 11), min(13, 15), max(13, 15)]
                    // ]
                }

                // Reshape/Merge it back into 2-D from the 3-D we created for performing
                // the Min/Max compare and switches
                global_chunks_a = g.reshape(
                    intermediate_chunks_a[intermediate_chunks_a.len() - 1].clone(),
                    array_type(vec![num_classes, num_class_reps, self.b], BIT),
                )?;
                // For it==1, i==0, reshape latest intermediate_chunk_a from [2, 1, 8, x] -> [2, 8, x] for next global_chunks_a
                // global_chunks_a:
                //  [
                //    [min(0, 1), max(0, 1), min(2, 3), max(2, 3), min(4, 5), max(4, 5), min(6, 7), max(2, 7)],
                //    [min(8, 9), max(8, 9), min(10, 11), max(10, 11), min(12, 13), max(12, 13), min(14, 15), max(14, 15)]
                //  ]

                // For it==2, i==1, reshape latest intermediate_chunk_a from [2, 2, 4, x] -> [4, 4, x] for next global_chunks_a
                // global_chunks_a:
                //  [
                //      [min(0, 2), max(0, 2), min(4, 6), max(4, 6)],
                //      [min(8, 10), max(8, 10), min(12, 14), max(12, 14)],
                //      [min(1, 3), max(1, 3), min(5, 7), max(5, 7)],
                //      [min(9, 11), max(9, 11), min(13, 15), max(13, 15)]
                //  ]
            }

            // Permute axes to revert original transpose
            let aa_transposed = g.permute_axes(global_chunks_a.clone(), vec![1, 0, 2])?;
            // For it==1, i==0, aa_transposed shape: [8, 2, x], with X-axis representing the classes
            // aa_transposed:
            // [
            //  [min(0, 1), max(0, 1)],
            //  [min(2, 3), max(2, 3)],
            //  [min(4, 5), max(4, 5)],
            //  [min(6, 7), max(2, 7)],
            //  [min(8, 9), max(8, 9)],
            //  [min(10, 11), max(10, 11)],
            //  [min(12, 13), max(12, 13)],
            //  [min(14, 15), max(14, 15)]
            // ]

            // Reshape data to flatten into shape [n, x] for further processing
            stage_ops.push(g.reshape(aa_transposed, array_type(vec![n, self.b], BIT))?)
            // In terms of the initial index positions of elements, this looks like:

            // data idx:          0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
            // data:           [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85]
            // For it==1, i==0, aa_transposed ==
            // idx:                  0         1          2          3          4          5          6          7          8          9         10           11           12           13           14           15
            // Current round idx: [min(0, 1), max(0, 1), min(2, 3), max(2, 3), min(4, 5), max(4, 5), min(6, 7), max(6, 7), min(8, 9), max(8, 9), min(10, 11), max(10, 11), min(12, 13), max(12, 13), min(14, 15), max(14, 15)]
            // data idx:          0    1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
            // data post ops.:  [99, 100, 97, 98, 95, 96, 93, 94, 91, 92, 89, 90, 87, 88, 85, 86]
        }
        stage_ops[self.k as usize].set_as_output()?;
        g.finalize()
    }

    fn get_name(&self) -> String {
        format!(
            "Sort(k={}, b={}, signed_comparison={})",
            self.k, self.b, self.signed_comparison
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom_ops::run_instantiation_pass;
    use crate::data_types::{scalar_size_in_bits, ScalarType, BIT, INT64, UINT16, UINT32, UINT64};
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
        let graph = context.create_graph()?;
        let n = 2u64.pow(k);
        let b = scalar_size_in_bits(st.clone());
        let i = graph.input(array_type(vec![n], st.clone()))?;
        let i_binary = if st == BIT {
            i.reshape(array_type(vec![n, 1], BIT))?
        } else {
            i.a2b()?
        };
        let signed_comparison = st.get_signed();
        let sorted = graph.custom_op(
            CustomOperation::new(Sort {
                k,
                b,
                signed_comparison,
            }),
            vec![i_binary],
        )?;
        let o = if st == BIT {
            sorted
        } else {
            sorted.b2a(st.clone())?
        };
        o.set_as_output()?;
        graph.finalize()?;
        graph.set_as_main()?;
        context.finalize()?;

        let mapped_c = run_instantiation_pass(graph.get_context())?;

        let seed = b"\xB6\xD7\x1A\x2F\x88\xC1\x12\xBA\x3F\x2E\x17\xAB\xB7\x46\x15\x9A";
        let mut prng = PRNG::new(Some(seed.clone()))?;
        let array_t: Type = array_type(vec![n], st.clone());
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
        let graph = context.create_graph()?;
        let n = 2u64.pow(k);
        let b = scalar_size_in_bits(st.clone());
        let signed_comparison = st.get_signed();
        let i = graph.input(array_type(vec![n], st.clone()))?;
        let i_binary = if st == BIT {
            i.reshape(array_type(vec![n, 1], BIT))?
        } else {
            i.a2b()?
        };
        let sorted = graph.custom_op(
            CustomOperation::new(Sort {
                k,
                b,
                signed_comparison,
            }),
            vec![i_binary],
        )?;
        let o = if st == BIT {
            sorted
        } else {
            sorted.b2a(st.clone())?
        };
        o.set_as_output()?;
        graph.finalize()?;
        graph.set_as_main()?;
        context.finalize()?;

        let mapped_c = run_instantiation_pass(graph.get_context())?;

        let v_a = Value::from_flattened_array(&data, st.clone())?;
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
