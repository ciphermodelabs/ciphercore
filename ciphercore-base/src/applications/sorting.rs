use crate::custom_ops::CustomOperation;
use crate::data_types::{array_type, vector_type, Type, BIT};
use crate::errors::Result;
use crate::graphs::SliceElement::SubArray;
use crate::graphs::*;
use crate::ops::min_max::{Max, Min};

/// This function returns a graph for a sorting network based on Batcher's
/// algorithm - \[b\] <https://math.mit.edu/~shor/18.310/batcher.pdf>
/// - Supports input bit array of shape (2^{k}, b), where 2^{k} elements are
/// represented as 'b' number of bits
pub fn create_batchers_sorting_graph(context: Context, k: u32, b: u64) -> Result<Graph> {
    match b {
        1 | 8 | 16 | 32 | 64 => {}
        _ => {
            return Err(runtime_error!(
                "Unsupported input bit size (b)! Supports only 8, 16, 32, or 64!"
            ));
        }
    }

    // NOTE: The implementation is based on the 'bottom up' approach as described in
    // the lecture note [b] cited above.
    // -Commenting about the initial few shape changes done with the help of a
    // 16 element array example
    let n = 2_u64.pow(k);
    let b_graph = context.create_graph()?;
    let i_a = b_graph.input(Type::Array(vec![n, b], BIT))?;
    let mut stage_ops = vec![i_a];
    // - The following loop, over 'it', corresponds to sorting (SORT()) operation
    // in [b]
    for it in 1..(k + 1) {
        let num_classes: u64 = 2_u64.pow(it);
        let num_class_reps = n / num_classes;
        let data_to_sort = stage_ops[(it - 1) as usize].clone();
        // for it==1, we are sorting into pairs i.e. we will have pairs of sorted keys
        // for it==2, we are creating sorted groups of size 4
        // for it==3, we are creating sorted groups of size 8
        // for it==4, we are creating sorted groups of size 16 and so on

        // For the purposes of the discussion, we will temporarily disregard the
        // final dimension i.e. the bit dimension so as to understand how the
        // jiggling of array shape is happening for the elements involved

        // Divide the keys into multiple i.e. 2^{it} classes or groups
        let global_a_reshape = b_graph.reshape(
            data_to_sort.clone(),
            array_type(vec![num_class_reps, num_classes, b], BIT),
        )?;

        // 1-D Array Indices: ->                  0  1  2  3      14 15
        // <At it==1> you would have 2^1 classes: 0, 1, 0, 1, ..., 0, 1
        // Now, global_a_reshape shape (2-D shape), in terms of indices, looks like:
        // class0|  class1|
        // ______|________|
        //      0|       1|
        //      2|       3|
        //      .|       .|
        //      .|       .|
        //     12|      13|
        //     14|      15|

        //  1-D Array Indices: ->                 0  1  2  3  4  5      10 11 12 13 14 15
        // <at it==2> you would have 2^2 classes: 0, 1, 2, 3, 0, 1, ..., 2, 3, 0, 1, 2, 3
        // Now, 2-D global_a_reshape shape, in terms of indices, looks like:
        // class0  class1  class2  class3
        //      0       1       2       3
        //      4       5       6       7
        //      8       9      10      11
        //      12     13      14      15

        //  1-D Array Indices:   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
        // <at it==3> 8 classes: 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7
        // Now, 2-D global_a_reshape shape, in terms of indices, looks like:
        // class0  class1  class2  class3 class4  class5  class6  class7
        //      0       1       2       3      4       5       6       7
        //      8       9      10      11     12      13      14      15

        // Permute the axes to perform the transpose operation
        // This is done so that each row now corresponds to a single class or group
        let mut global_chunks_a = b_graph.permute_axes(global_a_reshape, vec![1, 0, 2])?;
        // based on 'it' the global_chunks_a shape looks like
        // < For it == 1 >, locations of flat (1-D) indices
        // class0:  0, 2, ..., 12, 14
        // class1:  1, 3, ..., 13, 15

        // < For it == 2 >, locations of flat (1-D) indices
        // class0:  [[0, 4, 8, 12]
        // class1:   [1, 5, 9, 13]
        // class2:   [2, 6, 10, 14]
        // class3:   [3, 7, 11, 15]]

        // < For it == 3 >, locations of flat (1-D) indices
        // class0:  [[0, 8]
        // class1:   [1, 9]
        // class2:   [2, 10]
        // class3:   [3, 11]
        // class4:   [4, 12]
        // class5:   [5, 13]
        // class6:   [6, 14]
        // class7:   [7, 15]]

        let mut intermediate_chunks_a: Vec<Node> = vec![];
        // - The below loop, over 'i', corresponds to the MERGE() operation in [b]
        // - In the 'bottom up' approach, the operations contained in loop are
        // also referred as 'round(s) of comparisons' in [b]
        // - For groups or classes of size 2^{it}, you would require 'it' rounds
        // of comparisons
        // - The operations are vectorized to leverage the inherent parallelism
        // - For each group or class to be sorted, intially pairs are formed for
        // sorting then groups of 4 are formed for sorting, likewise for 8, 16 and so on.
        // - Technically, here, the number of dimensions are 4, however, we will
        // ignore the innermost dimension that corresponds to bits as it would
        // be handled by the custom_ops Min{} and Max{} and is not as relevant
        // to the Batcher's algorithm logic
        // - Formation of sub-groups of key sizes 2 or 4, 8, 16, ... for each group
        // of size 2^{it} happens along the outermost axis, whose size is
        // referenced here by 'chunks_a_sz_z'
        for i in (0..it).rev() {
            let chunks_a_sz_y = 2_u64.pow(i);
            let chunks_a_sz_z = 2_u64.pow(it - i); //n / (chunks_a_sz_y * num_class_reps);

            // - Reshape to create an additional dimension which corresponds to
            // each sub-group (of sizes 2, 4, 8, 16, ..., 2^{it-i}) within
            // original group of 2^{it} keys, which is to be sorted
            let chunks_a = b_graph.reshape(
                global_chunks_a.clone(),
                array_type(vec![chunks_a_sz_z, chunks_a_sz_y, num_class_reps, b], BIT),
            )?;

            // <For it==1 and i==0>,
            // - The two sub-groups are placed side-by-side along for sorting pairs
            // the outermost (Z) axis, Y-axis (height) is 1 and X-axis (breadth) is 8
            // i.e. Z_0 corresponds to class0, Z_1 corresponds to class1 and so on

            // <For it==2 and i==1>,
            // Sorting the groups of 4 by first sorting the pairs within them
            // chunks_a:
            // [ [[0, 4, 8, 12],    Values    [ [[min(0, 1), min(4, 5), min(8, 9), min(12, 13)],
            //    [1, 5, 9, 13]],   =====>       [max(0, 1), max(4, 5), max(8, 9)], max(12, 13)],
            //   [[2, 6, 10, 14],               [[min(2, 3), min(6, 7), min(10, 11), min(14, 15)],
            //    [3, 7, 11, 15]]                [max(2, 3), max(2, 7), max(10, 11), max(14, 15)]]
            // ]                              ]
            //
            // <For it==2 and i==0>,
            // Sorting the groups of 4 by sorting all 4 elements
            // i.e. Z_0 corresponds to class0, Z_1 corresponds to class1 and so on
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
            let single_first_element = b_graph.get(chunks_a.clone(), vec![0])?;
            // for it==1, i==0, single_first_element shape would be [1, 8, x]
            // for it==2, i==1, single_first_element shape would be [2, 4, x]
            // for it==2, i==0, single_first_element shape would be [1, 4, x]

            // If first step, then arrange odd-even adjacent pairs of keys into ordered pairs
            if i == it - 1 {
                // code to sort the only two chunks/halfs only

                // Here, we are dealing with just two classes: Z_0 (odds) and Z_1 (evens)
                // For it==1, i==0,
                // - Z_0 and Z_1 shapes are [1, 8, x]

                // For it==2, i==1,
                // - Z_0 and Z_1 shapes are [2, 4, x]

                // Get the group of odd indexed keys from each group or class i.e. Z_{0}
                let uu = single_first_element;

                // Get the group of even indexed keys from each group or class i.e. Z_{1}
                let vv = b_graph.get(chunks_a.clone(), vec![1])?;

                // Get minimums from both the classes
                let chunks_a_0 = b_graph
                    .custom_op(CustomOperation::new(Min {}), vec![uu.clone(), vv.clone()])?;
                // for it==1, i==0, chunks_a_0 = [[min(0, 1), min(2, 3), ..., min(12, 13), min(14, 15)]]
                // for it==2, i==1, chunks_a_0 = [[min(0, 2), min(4, 6), min(8, 10), min(12, 14)],
                //                                [min(1, 3), min(5, 7), min(9, 11), min(13, 15)]]

                // Get maximums from both the classes
                let chunks_a_1 = b_graph
                    .custom_op(CustomOperation::new(Max {}), vec![uu.clone(), vv.clone()])?;
                // for it==1, i==0, chunks_a_1 = [[max(0, 1), max(2, 3), ..., max(12, 13), max(14, 15)]]
                // for it==2, i==1, chunks_a_0 = [[max(0, 2), max(4, 6), max(8, 10), max(12, 14)],
                //                                [max(1, 3), max(5, 7), max(9, 11), max(13, 15)]]

                // Collect these maximums and minimums together for reshaping later
                let a_combined = b_graph.create_tuple(vec![chunks_a_0, chunks_a_1])?;
                // for it==1, i==0,
                // a_combined = [(min(0, 1), max(0, 1)), (min(2, 3), max(2, 3)), ..., (min(12, 13), max(12, 13)), (min(14, 15), max(14, 15))]
                // for it==2, i==1,
                // a_combined = [[(min(0, 2), max(0, 2)), (min(4, 6), max(4, 6)), (min(8, 10), max(8, 10)), (min(12, 14), max(12, 14))],
                //               [(min(1, 3), max(1, 3)), (min(5, 7), max(5, 7)), (min(9, 11), max(9, 11)), (min(13, 15), max(13, 15))]]

                // Reshape these combined elements back into a vector shape
                let interm_chunks_a = b_graph.reshape(
                    a_combined,
                    vector_type(
                        chunks_a_sz_z,
                        array_type(vec![chunks_a_sz_y, num_class_reps, b], chunks_a_scalar_t),
                    ),
                )?;
                // for it==1, i==0,
                // i.e. chunks_a's shape [2, 1, 8, x] for further processing
                // interm_chunks_a =    <[min(0, 1), max(0, 1), min(2, 3), max(2, 3), min(4, 5), max(4, 5), min(6, 7), max(6, 7)]>,
                //                      <[min(8, 9), max(8, 9), min(10, 11), max(10, 11), min(12, 13), max(12, 13), min(14, 15), max(14, 15)]>

                // for it==2, i==1,
                // chunks_a's shape [2, 2, 4, x]
                // interm_chunks_a = <[ [min(0, 2), max(0, 2), min(4, 6), max(4, 6)],
                //                      [min(8, 10), max(8, 10), min(12, 14), max(12, 14)] ],
                //                    [ [min(1, 3), max(1, 3), min(5, 7), max(5, 7)],
                //                      [min(9, 11), max(9, 11), min(13, 15), max(13, 15)] ]>

                // V2A these combined elements back into original shape
                intermediate_chunks_a.push(b_graph.vector_to_array(interm_chunks_a)?);
                // for it==1, i==0
                // i.e. into the chunks_a's shape [2, 1, 8, x] for further processing
                // intermediate_chunks_a[intermediate_chunks_a.len()-1] =
                // [
                //  [[min(0, 1), max(0, 1), min(2, 3), max(2, 3), min(4, 5), max(4, 5), min(6, 7), max(2, 7)]],
                //
                //  [[min(8, 9), max(8, 9), min(10, 11), max(10, 11), min(12, 13), max(12, 13), min(14, 15), max(14, 15)]]
                // ]

                // for it==2, i==1
                // i.e. into the chunks_a's shape [2, 2, 4, x] for further processing
                // intermediate_chunks_a[intermediate_chunks_a.len()-1] =
                //  [
                //      [[min(0, 2), max(0, 2), min(4, 6), max(4, 6)],
                //       [min(8, 10), max(8, 10), min(12, 14), max(12, 14)]],
                //      [[min(1, 3), max(1, 3), min(5, 7), max(5, 7)],
                //       [min(9, 11), max(9, 11), min(13, 15), max(13, 15)]]
                //  ]
            } else {
                // - This else block corresponds to the COMP() operations
                // specified within the MERGE() function in ([b], pg. 3) and
                // if x_{1}, x_{2}, ..., x_{n} are the keys to be sorted then
                // this COMP is operated as COMP(x 2 , x 3 ), COMP(x 4 , x 5 ), · · ·
                // COMP(x n−2 , x n−1 ).
                // In this case, we would not be considering terminal sub-groups
                // i.e. Z_{0} and Z_{2^{it-i}-1}

                // Set the shape of Z_0
                let a_single_first_elem = b_graph.reshape(
                    single_first_element,
                    array_type(
                        chunks_a_shape[1..chunks_a_shape.len()].to_vec(),
                        chunks_a_scalar_t.clone(),
                    ),
                )?;
                // for it==2, i==0,
                // a_single_first_elem =
                // [min(0, 2), max(0, 2), min(4, 6), max(4, 6)]

                // Obtain all the odd components of Z, except the first and last one i.e.
                // Z_{i} s.t. 1 <=i < 2^{it-i}-1 && i % 2 == 1
                let uu = b_graph
                    .get_slice(chunks_a.clone(), vec![SubArray(Some(1), Some(-1), Some(2))])?;
                // for it==2, i==0, uu shape = [1, 1, 4, x], uu =
                // [
                //      [[min(8, 10), max(8, 10), min(12, 14), max(12, 14)]],
                // ]

                // Obtain all the even components of Z, except the first one i.e.
                // Z_{i} s.t. 2 <= i < 2^{it-i} && i % 2 == 0
                let vv =
                    b_graph.get_slice(chunks_a.clone(), vec![SubArray(Some(2), None, Some(2))])?;
                // for it==2, i==0, vv shape = [1, 1, 4, x], vv =
                // [
                //      [[min(1, 3), max(1, 3), min(5, 7), max(5, 7)]],
                // ]

                // Obtain the minimum of these two arrays - uu and vv
                let chunks_a_evens = b_graph
                    .custom_op(CustomOperation::new(Min {}), vec![uu.clone(), vv.clone()])?;
                // for it==2, i==0, chunks_a_evens shape = [1, 1, 4, x], chunks_a_evens =
                // [
                //      [[min(8, 10, 1, 3), min(max(8, 10), max(1, 3)), min(12, 14, 5, 7), min(max(12, 14), max(5, 7))]]
                // ]

                // Obtain the maximum of these two arrays - uu and vv
                let chunks_a_odds = b_graph
                    .custom_op(CustomOperation::new(Max {}), vec![uu.clone(), vv.clone()])?;
                // for it==2, i==0, chunks_a_odds shape = [1, 1, 4, x], chunks_a_odds =
                // [
                //      [[max(min(8, 10), min(1, 3)), max(8, 10, 1, 3), max(min(12, 14), min(5, 7)), max(12, 14, 5, 7)]]
                // ]

                // Convert the array to vector and remove the extra Z-dimension
                let v_non_terminal_evens = b_graph.array_to_vector(chunks_a_evens)?;
                // for it==2, i==0, v_non_terminal_evens shape = [1, 4, x]<1>
                // v_non_terminal_evens =
                // <[min(8, 10, 1, 3), min(max(8, 10), max(1, 3)), min(12, 14, 5, 7), min(max(12, 14), max(5, 7))]>

                // Convert the array to vector and remove the extra Z-dimension
                let v_non_terminal_odds = b_graph.array_to_vector(chunks_a_odds)?;
                // for it==2, i==0, v_non_terminal_odds shape = [1, 4, x]<1>
                // v_non_terminal_odds =
                // <[max(min(8, 10), min(1, 3)), max(8, 10, 1, 3), max(min(12, 14), min(5, 7)), max(12, 14, 5, 7)]>

                // Zip both the results together
                let v_non_term_elems =
                    b_graph.zip(vec![v_non_terminal_evens, v_non_terminal_odds])?;
                // for it==2, i==0, v_non_term_elems shape = ((1, 4, x)(1, 4, x))<1>
                // v_non_term_elems =
                // <(min(8, 10, 1, 3), max(min(8, 10), min(1, 3))),
                //  (min(max(8, 10), max(1, 3)), max(8, 10, 1, 3)),
                //  (min(12, 14, 5, 7), max(min(12, 14), min(5, 7))),
                //  (min(max(12, 14), max(5, 7)), max(12, 14, 5, 7))>

                // In a similar way to the first element i.e. Z_{0}, extract
                // the last element
                let single_last_elem =
                    b_graph.get(chunks_a.clone(), vec![chunks_a_shape[0] - 1])?;
                // for it==2, i==0, single_last_element shape would be [1, 4, x]

                // Set the shape of Z_{2^{it-i}-1} to [1, 4, x]
                let a_single_last_elem = b_graph.reshape(
                    single_last_elem,
                    array_type(
                        chunks_a_shape[1..chunks_a_shape.len()].to_vec(),
                        chunks_a_scalar_t.clone(),
                    ),
                )?;
                // for it==2, i==0,
                // a_single_last_elem =
                //  [min(9, 11), max(9, 11), min(13, 15), max(13, 15)]

                // Create a tuple of Z: (first element-Z_{0}, vector, last element-Z_{2^{it-i}-1})
                let v_combined = b_graph.create_tuple(vec![
                    a_single_first_elem,
                    v_non_term_elems,
                    a_single_last_elem,
                ])?;
                // for it==2, i==0
                // v_combined =
                // ([min(0, 2), max(0, 2), min(4, 6), max(4, 6)],
                //  <(min(8, 10, 1, 3), max(min(8, 10), min(1, 3))),
                //  (min(max(8, 10), max(1, 3)), max(8, 10, 1, 3)),
                //  (min(12, 14, 5, 7), max(min(12, 14), min(5, 7))),
                //  (min(max(12, 14), max(5, 7)), max(12, 14, 5, 7))>,
                //  [min(9, 11), max(9, 11), min(13, 15), max(13, 15)]
                // )

                // Reshape the tuple back into vector form
                let v_chunk_a = b_graph.reshape(
                    v_combined,
                    vector_type(
                        chunks_a_shape[0],
                        array_type(
                            chunks_a_shape[1..chunks_a_shape.len()].to_vec(),
                            chunks_a_scalar_t,
                        ),
                    ),
                )?;
                // for it==2, i==0,
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

                // Convert the vector form to array form
                intermediate_chunks_a.push(b_graph.vector_to_array(v_chunk_a)?);
                // for it==2, i==0
                // intermediate_chunks_a would be
            }

            // Reshape/Merge it back into 2-D from the 3-D we created for performing
            // the Min/Max compare and switches
            global_chunks_a = b_graph.reshape(
                intermediate_chunks_a[(intermediate_chunks_a.len() - 1) as usize].clone(),
                array_type(vec![num_classes, num_class_reps, b], BIT),
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

        // Permute_axes: Perform transpose to revert original transpose
        let aa_transposed = b_graph.permute_axes(global_chunks_a.clone(), vec![1, 0, 2])?;
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
        stage_ops.push(b_graph.reshape(aa_transposed, array_type(vec![n, b], BIT))?)
        // In terms of the initial index positions of elements, this looks like:

        // data idx:          0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
        // data:           [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85]
        // <it==1, i==0>: aa_transposed ==
        // idx:                  0         1          2          3          4          5          6          7          8          9         10           11           12           13           14           15
        // Current round idx: [min(0, 1), max(0, 1), min(2, 3), max(2, 3), min(4, 5), max(4, 5), min(6, 7), max(6, 7), min(8, 9), max(8, 9), min(10, 11), max(10, 11), min(12, 13), max(12, 13), min(14, 15), max(14, 15)]
        // data idx:          0    1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
        // data post ops.:  [99, 100, 97, 98, 95, 96, 93, 94, 91, 92, 89, 90, 87, 88, 85, 86]
    }
    b_graph.set_output_node(stage_ops[k as usize].clone())?;
    b_graph.finalize()?;

    Ok(b_graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom_ops::run_instantiation_pass;
    use crate::data_types::{ScalarType, UINT16, UINT32, UINT64, UINT8};
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::random::PRNG;
    use std::cmp::Reverse;

    /// Helper function to test the sorting network graph for large inputs
    /// - Inputs -
    /// k: Parameter for input size (2^{k})
    /// b: Number of bits used for representing individual the array element
    /// - Testing is done by first sorting it with the given graph and then
    /// comparing its result with the non-graph-sorted result.
    fn test_large_vec_unsigned_batchers_sorting(k: u32, b: u64) -> Result<()> {
        let context = create_context()?;
        let graph: Graph = create_batchers_sorting_graph(context.clone(), k, b)?;
        context.set_main_graph(graph.clone())?;
        context.finalize()?;

        let mapped_c = run_instantiation_pass(graph.get_context())?;
        let s_type: ScalarType = match b {
            1 => BIT,
            8 => UINT8,
            16 => UINT16,
            32 => UINT32,
            64 => UINT64,
            _ => {
                return Err(runtime_error!("Unsupported bit size"));
            }
        };

        let seed = b"\xB6\xD7\x1A\x2F\x88\xC1\x12\xBA\x3F\x2E\x17\xAB\xB7\x46\x15\x9A";
        let mut prng = PRNG::new(Some(seed.clone()))?;
        let array_t: Type = array_type(vec![2_u64.pow(k)], s_type);
        let data = prng.get_random_value(array_t.clone())?;
        let data_v_u64 = data.to_flattened_array_u64(array_t.clone())?;
        let result = random_evaluate(mapped_c.mappings.get_graph(graph), vec![data])?
            .to_flattened_array_u64(array_t)?;
        let mut sorted_data = data_v_u64;
        sorted_data.sort_unstable();
        assert_eq!(sorted_data, result);
        Ok(())
    }

    /// Helper function to test the sorting network graph
    /// - Inputs -
    /// k: Parameter for input size (2^{k})
    /// b: Number of bits used for representing individual the array element
    /// data: Sample data points
    /// - Testing is done by first sorting it with the given graph and then
    /// comparing its result with the non-graph-sorted result.
    fn test_unsigned_batchers_sorting_graph_helper(k: u32, b: u64, data: Vec<u64>) -> Result<()> {
        let context = create_context()?;
        let graph: Graph = create_batchers_sorting_graph(context.clone(), k, b)?;
        context.set_main_graph(graph.clone())?;
        context.finalize()?;

        let mapped_c = run_instantiation_pass(graph.get_context())?;
        let s_type: ScalarType = match b {
            1 => BIT,
            8 => UINT8,
            16 => UINT16,
            32 => UINT32,
            64 => UINT64,
            _ => {
                return Err(runtime_error!("Unsupported bit size"));
            }
        };

        let v_a = Value::from_flattened_array(&data, s_type.clone())?;
        let result = random_evaluate(mapped_c.mappings.get_graph(graph), vec![v_a])?
            .to_flattened_array_u64(array_type(vec![data.len() as u64], s_type))?;
        let mut sorted_data = data;
        sorted_data.sort_unstable();
        assert_eq!(sorted_data, result);
        Ok(())
    }

    /// - This function tests the well-formed sorting graph for its correctness
    /// - Parameters varied are k, b and the input data could be unsorted,
    /// sorted or sorted in a decreasing order.
    #[test]
    fn test_wellformed_unsigned_batchers_sorting_graph() -> Result<()> {
        let mut data = vec![65535, 0, 2, 32768];
        test_unsigned_batchers_sorting_graph_helper(2, 16, data.clone())?;
        data.sort_unstable();
        test_unsigned_batchers_sorting_graph_helper(2, 16, data.clone())?;
        data.sort_by_key(|w| Reverse(*w));
        test_unsigned_batchers_sorting_graph_helper(2, 16, data.clone())?;

        let data = vec![548890456, 402403639693304868, u64::MAX, 999790788];
        test_unsigned_batchers_sorting_graph_helper(2, 64, data.clone())?;

        let data = vec![643082556];
        test_unsigned_batchers_sorting_graph_helper(0, 32, data.clone())?;

        let data = vec![1, 0, 0, 1];
        test_unsigned_batchers_sorting_graph_helper(2, 1, data.clone())?;

        test_large_vec_unsigned_batchers_sorting(7, 1)?;
        test_large_vec_unsigned_batchers_sorting(4, 64)?;

        Ok(())
    }

    /// This function tests if the bad input parameters are detected and if the
    /// error is generated for them.
    #[test]
    fn test_malformed_unsigned_batchers_sorting_graph() -> Result<()> {
        {
            let context = create_context()?;
            assert!(create_batchers_sorting_graph(context.clone(), 2, 0).is_err());
        }
        {
            let context = create_context()?;
            assert!(create_batchers_sorting_graph(context.clone(), 2, 63).is_err());
        }
        {
            let context = create_context()?;
            assert!(create_batchers_sorting_graph(context.clone(), 2, 128).is_err());
        }
        Ok(())
    }
}
