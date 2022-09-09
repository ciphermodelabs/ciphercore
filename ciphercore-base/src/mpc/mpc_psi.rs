use crate::custom_ops::CustomOperationBody;
use crate::data_types::{array_type, vector_type, Type, BIT, UINT64};
use crate::errors::Result;
use crate::graphs::{Context, Graph};
use crate::ops::utils::{pull_out_bits, put_in_bits, zeros};

use serde::{Deserialize, Serialize};

/// Adds a node returning hash values of an input array of binary strings using provided hash functions.
///
/// Hash functions are defined as an array of binary matrices.
/// The hash of an input string is a product of one of these matrices and this string.
/// Hence, the last dimension of these matrices should coincide with the length of input strings.
///
/// If the input array has shape `[..., n, b]` and hash matrices are given as an `[h, m, b]`-array,
/// then the hash map is an array of shape `[..., h, n]`.
/// The hash table element with index `[..., h, i]` is equal to `j` if the `[..., i]`-th `b`-bit input string is hashed to `j` by the `h`-th hash function.
///
/// When used within a PSI protocol, the hash functions should be the same as those used for Cuckoo hashing.    
///
/// **WARNING**: this function should not be used before MPC compilation.
///
/// # Custom operation arguments
///
/// - input array of binary strings of shape [..., n, b]
/// - random binary [h, m, b]-matrix.
///
/// # Custom operation returns
///
/// hash table of shape [..., h, n] containing UINT64 elements
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
struct SimpleHash;

#[typetag::serde]
impl CustomOperationBody for SimpleHash {
    fn instantiate(&self, context: Context, argument_types: Vec<Type>) -> Result<Graph> {
        if argument_types.len() != 2 {
            // Panics since:
            // - the user has no direct access to this function.
            // - the MPC compiler should pass the correct number of arguments
            // and this panic should never happen.
            panic!("SimpleHash should have 2 inputs.");
        }

        let input_type = argument_types[0].clone();
        let hash_type = argument_types[1].clone();

        if !matches!(input_type, Type::Array(_, BIT)) {
            return Err(runtime_error!(
                "SimpleHash can't be applied to a non-binary arrays"
            ));
        }
        let input_shape = input_type.get_shape();
        if input_shape.len() < 2 {
            return Err(runtime_error!(
                "Input shape must have at least 2 dimensions"
            ));
        }
        if !matches!(hash_type, Type::Array(_, BIT)) {
            return Err(runtime_error!(
                "SimpleHash needs a binary array as a hash matrix"
            ));
        }
        let hash_shape = hash_type.get_shape();
        if hash_shape.len() != 3 {
            return Err(runtime_error!("Hash array should have 3 dimensions"));
        }
        if hash_shape[1] > 63 {
            return Err(runtime_error!(
                "Hash map is too big. Decrease the number of rows of hash matrices"
            ));
        }
        let input_element_length = input_shape[input_shape.len() - 1];
        if hash_shape[2] != input_element_length {
            return Err(runtime_error!(
                "Hash matrix accepts bitstrings of length {}, but input strings are of length {}",
                hash_shape[2],
                input_element_length
            ));
        }

        let g = context.create_graph()?;

        let input_array = g.input(input_type.clone())?;
        let hash_matrices = g.input(hash_type.clone())?;

        let hash_shape = hash_type.get_shape();

        let mut extended_shape = input_type.get_shape();
        extended_shape.insert(extended_shape.len() - 1, 1);

        // For each subarray and for each hash function, the output hash map contains hashes of input bit strings
        let input_shape = input_type.get_shape();
        let mut single_hash_table_shape = input_shape[0..input_shape.len() - 1].to_vec();
        single_hash_table_shape.push(hash_shape[1]);

        // Multiply hash matrices of shape [h, m, b] by input strings of shape [..., n, b].
        // In Einstein notation, ...nb, hmb -> ...hnm.

        // Change the shape of hash_matrices from [h,m,b] to [b, h*m]
        let hash_matrices_for_matmul = hash_matrices
            .reshape(array_type(
                vec![hash_shape[0] * hash_shape[1], hash_shape[2]],
                BIT,
            ))?
            .permute_axes(vec![1, 0])?;

        // The result shape is [..., n, h*m]
        let mut hash_tables = input_array.matmul(hash_matrices_for_matmul)?;

        // Reshape to [..., n,  h, m]
        let mut split_by_hash_shape = input_shape[0..input_shape.len() - 1].to_vec();
        split_by_hash_shape.extend_from_slice(&hash_shape[0..2]);
        hash_tables = hash_tables.reshape(array_type(split_by_hash_shape.clone(), BIT))?;

        // Transpose to [..., h, n, m]
        let len_output_shape = split_by_hash_shape.len() as u64;
        let mut permuted_axes: Vec<u64> = (0..len_output_shape).collect();
        permuted_axes[len_output_shape as usize - 3] = len_output_shape - 2;
        permuted_axes[len_output_shape as usize - 2] = len_output_shape - 3;
        hash_tables = hash_tables.permute_axes(permuted_axes)?;

        hash_tables = pull_out_bits(hash_tables)?;
        let hash_suffix_type = hash_tables.get_type()?.get_shape()[1..].to_vec();
        let num_zeros = 64 - hash_shape[1];
        let zeros_type = vector_type(num_zeros, array_type(hash_suffix_type.clone(), BIT));
        let zeros = zeros(&g, zeros_type)?;

        hash_tables = g
            .create_tuple(vec![hash_tables.array_to_vector()?, zeros])?
            .reshape(vector_type(64, array_type(hash_suffix_type, BIT)))?
            .vector_to_array()?;

        hash_tables = put_in_bits(hash_tables)?.b2a(UINT64)?;

        hash_tables.set_as_output()?;

        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        "SimpleHash".to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::custom_ops::run_instantiation_pass;
    use crate::custom_ops::CustomOperation;
    use crate::data_types::scalar_type;
    use crate::data_types::ArrayShape;
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::graphs::create_context;

    fn simple_hash_helper(
        input_shape: ArrayShape,
        hash_shape: ArrayShape,
        inputs: Vec<Value>,
    ) -> Result<Vec<u64>> {
        let c = create_context()?;
        let g = c.create_graph()?;
        let i = g.input(array_type(input_shape.clone(), BIT))?;
        let hash_matrix = g.input(array_type(hash_shape.clone(), BIT))?;
        let o = g.custom_op(CustomOperation::new(SimpleHash), vec![i, hash_matrix])?;
        g.set_output_node(o)?;
        g.finalize()?;
        c.set_main_graph(g.clone())?;
        c.finalize()?;
        let mapped_c = run_instantiation_pass(c)?.context;
        let result_value = random_evaluate(mapped_c.get_main_graph()?, inputs)?;
        let mut result_shape = input_shape[0..input_shape.len() - 1].to_vec();
        result_shape.insert(0, hash_shape[0]);
        let result_type = array_type(result_shape, UINT64);
        result_value.to_flattened_array_u64(result_type)
    }

    fn simple_hash_helper_fails(input_t: Type, hash_t: Type) -> Result<()> {
        let c = create_context()?;
        let g = c.create_graph()?;
        let i = g.input(input_t)?;
        let hash_matrix = g.input(hash_t)?;
        let o = g.custom_op(CustomOperation::new(SimpleHash), vec![i, hash_matrix])?;
        g.set_output_node(o)?;
        g.finalize()?;
        c.set_main_graph(g.clone())?;
        c.finalize()?;
        run_instantiation_pass(c)?;
        Ok(())
    }

    #[test]
    fn test_simple_hash() {
        || -> Result<()> {
            // no collision
            {
                // [2,3]-array
                let input = Value::from_flattened_array(&[1, 0, 1, 0, 0, 1], BIT)?;
                // [3,2,3]-array
                let hash_matrix = Value::from_flattened_array(
                    &[1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1],
                    BIT,
                )?;
                // output [3,2]-array
                let expected = vec![0, 1, 0, 2, 3, 2];
                assert_eq!(
                    simple_hash_helper(vec![2, 3], vec![3, 2, 3], vec![input, hash_matrix])?,
                    expected
                );
            }
            // collisions
            {
                // [2,3]-array
                let input = Value::from_flattened_array(&[1, 0, 1, 0, 0, 0], BIT)?;
                // [3,2,3]-array
                let hash_matrix = Value::from_flattened_array(
                    &[1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1],
                    BIT,
                )?;
                // output [3,2]-array
                let expected = vec![0, 0, 0, 0, 3, 0];
                assert_eq!(
                    simple_hash_helper(vec![2, 3], vec![3, 2, 3], vec![input, hash_matrix])?,
                    expected
                );
            }
            {
                // [2,2,2]-array
                let input = Value::from_flattened_array(&[1, 0, 0, 0, 1, 1, 0, 1], BIT)?;
                // [2,3,2]-array
                let hash_matrix =
                    Value::from_flattened_array(&[1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1], BIT)?;
                // output [2,2,2]-array
                let expected = vec![3, 0, 2, 0, 7, 4, 7, 5];
                assert_eq!(
                    simple_hash_helper(vec![2, 2, 2], vec![2, 3, 2], vec![input, hash_matrix])?,
                    expected
                );
            }
            // malformed input
            {
                let input_t = scalar_type(BIT);
                let hash_t = array_type(vec![2, 3, 4], BIT);
                assert!(simple_hash_helper_fails(input_t, hash_t).is_err());
            }
            {
                let input_t = array_type(vec![5, 4], UINT64);
                let hash_t = array_type(vec![2, 3, 4], BIT);
                assert!(simple_hash_helper_fails(input_t, hash_t).is_err());
            }
            {
                let input_t = array_type(vec![4], BIT);
                let hash_t = array_type(vec![2, 3, 4], BIT);
                assert!(simple_hash_helper_fails(input_t, hash_t).is_err());
            }
            {
                let input_t = array_type(vec![5, 4], BIT);
                let hash_t = scalar_type(BIT);
                assert!(simple_hash_helper_fails(input_t, hash_t).is_err());
            }
            {
                let input_t = array_type(vec![5, 4], BIT);
                let hash_t = array_type(vec![3, 4], BIT);
                assert!(simple_hash_helper_fails(input_t, hash_t).is_err());
            }
            {
                let input_t = array_type(vec![5, 4], BIT);
                let hash_t = array_type(vec![2, 3, 4], UINT64);
                assert!(simple_hash_helper_fails(input_t, hash_t).is_err());
            }
            {
                let input_t = array_type(vec![5, 4], BIT);
                let hash_t = array_type(vec![2, 64, 4], BIT);
                assert!(simple_hash_helper_fails(input_t, hash_t).is_err());
            }
            {
                let input_t = array_type(vec![5, 4], BIT);
                let hash_t = array_type(vec![2, 3, 5], BIT);
                assert!(simple_hash_helper_fails(input_t, hash_t).is_err());
            }
            Ok(())
        }()
        .unwrap();
    }
}
