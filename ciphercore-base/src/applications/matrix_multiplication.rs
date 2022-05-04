use crate::data_types::{array_type, ScalarType};
use crate::errors::Result;
use crate::graphs::{Context, Graph};

/// Creates a graph that multiplies a matrix with integer entries of shape n x m by a matrix of shape m x k.
///
/// # Arguments
///
/// * `context` - context where a matrix multiplication graph should be created
/// * `n` - number of rows of the left matrix,
/// * `m` - number of columns of the left matrix (and number of rows of the right matrix)
/// * `k` - number of columns of the right matrix
/// * `st` - scalar type of matrix elements
///
/// # Returns
///
/// Graph that multiplies two matrices
pub fn create_matrix_multiplication_graph(
    context: Context,
    n: u64,
    m: u64,
    k: u64,
    st: ScalarType,
) -> Result<Graph> {
    // Create a graph in a given context that will be used for matrix multiplication
    let g = context.create_graph()?;

    // Create types of input matrices.
    // Matrices can be represented as arrays with two 2-dimensional shapes.
    // First, create the array type of a left matrix with shape `[n, m]`, which corresponds to a (n x m)-matrix.
    let left_matrix_type = array_type(vec![n, m], st.clone());
    // Second, create the array type of a right matrix with shape `[m, k]`, which corresponds to a (m x k)-matrix.
    let right_matrix_type = array_type(vec![m, k], st);

    // For each input matrix, add input nodes to the empty graph g created above.
    // Input nodes require the types of input matrices generated in previous lines.
    let left_matrix_input = g.input(left_matrix_type)?;
    let right_matrix_input = g.input(right_matrix_type)?;

    // Matrix multiplication is a built-in function of CipherCore, so it can be computed by a single computational node.
    let output = left_matrix_input.matmul(right_matrix_input)?;

    // Before computation, every graph should be finalized, which means that it should have a designated output node.
    // This can be done by calling `g.set_output_node(output)?` or as below.
    output.set_as_output()?;
    // Finalization checks that the output node of the graph g is set. After finalization the graph can't be changed.
    g.finalize()?;

    Ok(g)
}

#[cfg(test)]
mod tests {
    use crate::data_types::{INT32, INT64};
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::graphs::create_context;
    use std::ops::Not;

    use super::*;

    fn test_matmul_helper<
        T1: TryInto<u64> + Not<Output = T1> + TryInto<u8> + Copy,
        T2: TryInto<u64> + Not<Output = T2> + TryInto<u8> + Copy,
    >(
        input1_value: &[T1],
        input2_value: &[T2],
        n: u64,
        m: u64,
        k: u64,
        st: ScalarType,
    ) -> Value {
        || -> Result<Value> {
            let c = create_context()?;
            let g = create_matrix_multiplication_graph(c.clone(), n, m, k, st.clone())?;
            g.set_as_main()?;
            c.finalize()?;

            let left_matrix_type = array_type(vec![n, m], st.clone());
            let right_matrix_type = array_type(vec![m, k], st);
            let val1 =
                Value::from_flattened_array(input1_value, left_matrix_type.get_scalar_type())?;
            let val2 =
                Value::from_flattened_array(input2_value, right_matrix_type.get_scalar_type())?;
            random_evaluate(g, vec![val1, val2])
        }()
        .unwrap()
    }

    #[test]
    fn test_matmul() {
        || -> Result<()> {
            assert!(
                test_matmul_helper(
                    &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                    &[1, 2, 3, 4, 5, 6],
                    4,
                    3,
                    2,
                    INT32
                ) == Value::from_flattened_array(&[22, 28, 49, 64, 76, 100, 103, 136], INT32)?
            );
            assert!(
                test_matmul_helper(
                    &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                    &[6, 5, 4, 3, 2, 1],
                    4,
                    3,
                    2,
                    INT32
                ) == Value::from_flattened_array(&[20, 14, 56, 41, 92, 68, 128, 95], INT32)?
            );
            assert!(
                test_matmul_helper(
                    &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                    &[12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                    1,
                    12,
                    1,
                    INT32
                ) == Value::from_flattened_array(&[364], INT32)?
            );
            assert!(
                test_matmul_helper(
                    &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                    &[12, 45, 56],
                    4,
                    3,
                    1,
                    INT32
                ) == Value::from_flattened_array(&[270, 609, 948, 1287], INT32)?
            );
            assert!(
                test_matmul_helper(
                    &[12, 45, 56],
                    &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                    1,
                    3,
                    4,
                    INT32
                ) == Value::from_flattened_array(&[741, 854, 967, 1080], INT32)?
            );
            assert!(
                test_matmul_helper(
                    &[-1, 4, -3, -5, -6, 2],
                    &[-1, -2, 3, -4, -5, -6],
                    2,
                    3,
                    2,
                    INT32
                ) == Value::from_flattened_array(&[28, 4, -23, 22], INT32)?
            );
            assert!(
                test_matmul_helper(
                    &[-1, 4, -3, -5, -6, 2],
                    &[-1, -2, 3, -4, -5, -6],
                    2,
                    3,
                    2,
                    INT64
                ) == Value::from_flattened_array(&[28, 4, -23, 22], INT64)?
            );
            Ok(())
        }()
        .unwrap();
    }
}
