use crate::broadcast::{index_to_number, number_to_index};
use crate::bytes::{
    add_u64, add_vectors_u64, dot_vectors_u64, multiply_u64, multiply_vectors_u64,
    subtract_vectors_u64,
};
use crate::bytes::{vec_from_bytes, vec_to_bytes};
use crate::data_types::{array_type, ArrayShape, Type, BIT, UINT64};
use crate::data_values::Value;
use crate::errors::Result;
use crate::evaluators::Evaluator;
use crate::graphs::{Node, Operation};
use crate::random::{Prf, PRNG, SEED_SIZE};
use crate::slices::slice_index;
use crate::type_inference::transpose_shape;
use crate::typed_value::TypedValue;
use crate::typed_value_operations::TypedValueOperations;

use std::cmp::min;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::iter::repeat;

use super::join::evaluate_join;

/// It is assumed that shape can be broadcast to shape_res
fn broadcast_to_shape(arr: &[u64], shape: &[u64], shape_res: &[u64]) -> Vec<u64> {
    let res_length: u64 = shape_res.iter().product();
    let mut result = vec![];
    let offset = shape_res.len() - shape.len();
    for i in 0..res_length {
        let index_vec = number_to_index(i, shape_res);
        let index = index_to_number(&index_vec[offset..], shape);
        result.push(arr[index as usize]);
    }
    result
}

fn flatten_value(value: Value) -> Vec<Value> {
    value
        .access(
            |_| Ok(vec![value.clone()]),
            |vector| {
                Ok(vector
                    .iter()
                    .flat_map(|x| flatten_value(x.clone()))
                    .collect())
            },
        )
        .unwrap()
}

fn unflatten_value(flattened_value: &[Value], position: &mut u64, t: Type) -> Value {
    match t {
        Type::Scalar(_) | Type::Array(_, _) => {
            *position += 1;
            flattened_value[(*position - 1) as usize].clone()
        }
        Type::Tuple(vt) => {
            let mut result = vec![];
            for t1 in vt {
                result.push(unflatten_value(flattened_value, position, (*t1).clone()));
            }
            Value::from_vector(result)
        }
        Type::NamedTuple(vt) => {
            let mut result = vec![];
            for (_, t1) in vt {
                result.push(unflatten_value(flattened_value, position, (*t1).clone()));
            }
            Value::from_vector(result)
        }
        Type::Vector(len, t1) => {
            let mut result = vec![];
            for _ in 0..len {
                result.push(unflatten_value(flattened_value, position, (*t1).clone()));
            }
            Value::from_vector(result)
        }
    }
}

pub(crate) fn evaluate_add_subtract_multiply(
    type1: Type,
    value1: Value,
    type2: Type,
    value2: Value,
    operation: Operation,
    result_type: Type,
) -> Result<Value> {
    let result_bytes = match (type1.clone(), type2.clone()) {
        // scalar types and shapes will be compatible thanks to process_node
        (Type::Scalar(st), Type::Scalar(_))
        | (Type::Array(_, st), Type::Scalar(_))
        | (Type::Scalar(_), Type::Array(_, st))
        | (Type::Array(_, st), Type::Array(_, _)) => {
            //pack bytes into vectors of u64
            let bytes1_u64 = value1
                .access_bytes(|ref_bytes| Ok(vec_from_bytes(ref_bytes, st.clone())?.to_vec()))?;
            let bytes2_u64 = value2
                .access_bytes(|ref_bytes| Ok(vec_from_bytes(ref_bytes, st.clone())?.to_vec()))?;
            let shape1 = type1.get_dimensions();
            let shape2 = type2.get_dimensions();
            let shape_res = result_type.get_dimensions();
            let result_u64 = match operation {
                Operation::Add => add_vectors_u64(
                    &broadcast_to_shape(&bytes1_u64, &shape1, &shape_res),
                    &broadcast_to_shape(&bytes2_u64, &shape2, &shape_res),
                    st.get_modulus(),
                )?,
                Operation::Subtract => subtract_vectors_u64(
                    &broadcast_to_shape(&bytes1_u64, &shape1, &shape_res),
                    &broadcast_to_shape(&bytes2_u64, &shape2, &shape_res),
                    st.get_modulus(),
                )?,
                Operation::Multiply => multiply_vectors_u64(
                    &broadcast_to_shape(&bytes1_u64, &shape1, &shape_res),
                    &broadcast_to_shape(&bytes2_u64, &shape2, &shape_res),
                    st.get_modulus(),
                )?,
                _ => panic!("Should not be here"),
            };
            //unpack bytes from vectors of u64
            vec_to_bytes(&result_u64, st)?
        }
        _ => {
            return Err(runtime_error!("Not implemented"));
        }
    };
    let result_value = Value::from_bytes(result_bytes);
    Ok(result_value)
}

pub(crate) fn evaluate_mixed_multiply(
    type1: Type,
    value1: Value,
    type2: Type,
    value2: Value,
    result_type: Type,
) -> Result<Value> {
    let result_bytes = match (type1.clone(), type2.clone()) {
        // scalar types and shapes will be compatible thanks to process_node
        (Type::Scalar(st), Type::Scalar(_))
        | (Type::Array(_, st), Type::Scalar(_))
        | (Type::Scalar(st), Type::Array(_, _))
        | (Type::Array(_, st), Type::Array(_, _)) => {
            //pack bytes into vectors of u64
            let bytes1_u64 = value1
                .access_bytes(|ref_bytes| Ok(vec_from_bytes(ref_bytes, st.clone())?.to_vec()))?;
            let bytes2_u64 =
                value2.access_bytes(|ref_bytes| Ok(vec_from_bytes(ref_bytes, BIT)?.to_vec()))?;
            let shape1 = type1.get_dimensions();
            let shape2 = type2.get_dimensions();
            let shape_res = result_type.get_dimensions();
            let result_u64 = multiply_vectors_u64(
                &broadcast_to_shape(&bytes1_u64, &shape1, &shape_res),
                &broadcast_to_shape(&bytes2_u64, &shape2, &shape_res),
                st.get_modulus(),
            )?;
            //unpack bytes from vectors of u64
            vec_to_bytes(&result_u64, st)?
        }
        _ => {
            return Err(runtime_error!("Not implemented"));
        }
    };
    let result_value = Value::from_bytes(result_bytes);
    Ok(result_value)
}

fn evaluate_dot(
    type0: Type,
    value0: Value,
    type1: Type,
    value1: Value,
    result_type: Type,
) -> Result<Value> {
    let st = type0.get_scalar_type();
    let modulus = st.get_modulus();
    if type0.is_array() && type1.is_array() {
        let shape0 = type0.get_shape();
        let shape1 = type1.get_shape();
        let entries0 = value0.to_flattened_array_u64(type0)?;
        let entries1 = value1.to_flattened_array_u64(type1)?;
        let result_length = if result_type.is_scalar() {
            1
        } else {
            let result_shape = result_type.get_shape();
            result_shape.into_iter().product::<u64>() as usize
        };
        let mut result_entries = vec![0; result_length];
        if shape0.len() == 1 && shape1.len() == 1 {
            for i in 0..shape0[0] {
                result_entries[0] = add_u64(
                    result_entries[0],
                    multiply_u64(entries0[i as usize], entries1[i as usize], modulus),
                    modulus,
                );
            }
            Value::from_scalar(result_entries[0], st)
        } else {
            let result_shape = result_type.get_shape();
            let middle_dim = if shape1.len() > 1 {
                shape1[shape1.len() - 2]
            } else {
                shape1[0]
            };
            for i in 0..(result_entries.len() as u64) {
                let result_index = number_to_index(i, &result_shape);
                for j in 0..middle_dim {
                    let mut index0 = result_index[0..shape0.len() - 1].to_vec();
                    index0.push(j);
                    let mut index1: Vec<u64>;
                    if shape1.len() > 1 {
                        index1 = result_index[shape0.len() - 1..result_index.len()].to_vec();
                        index1.insert(index1.len() - 1, j);
                    } else {
                        index1 = vec![j];
                    }
                    let n0 = index_to_number(&index0, &shape0);
                    let n1 = index_to_number(&index1, &shape1);
                    result_entries[i as usize] = add_u64(
                        result_entries[i as usize],
                        multiply_u64(entries0[n0 as usize], entries1[n1 as usize], modulus),
                        modulus,
                    );
                }
            }
            Value::from_flattened_array(&result_entries, st)
        }
    } else {
        evaluate_add_subtract_multiply(
            type0,
            value0,
            type1,
            value1,
            Operation::Multiply,
            result_type,
        )
    }
}

fn evaluate_matmul(
    type0: Type,
    value0: Value,
    type1: Type,
    value1: Value,
    result_type: Type,
) -> Result<Value> {
    let st = type0.get_scalar_type();
    let modulus = st.get_modulus();
    if !type0.is_array() || !type1.is_array() {
        panic!("Inconsistency with type checker");
    }
    let mut shape0 = type0.get_shape();
    let mut shape1 = type1.get_shape();
    let entries0 = value0.to_flattened_array_u64(type0)?;
    let entries1 = value1.to_flattened_array_u64(type1)?;
    let result_length = if result_type.is_scalar() {
        1
    } else {
        let result_shape = result_type.get_shape();
        result_shape.into_iter().product::<u64>() as usize
    };
    let mut result_entries = vec![0; result_length];
    if shape0.len() == 1 && shape1.len() == 1 {
        for i in 0..shape0[0] {
            result_entries[0] = add_u64(
                result_entries[0],
                multiply_u64(entries0[i as usize], entries1[i as usize], modulus),
                modulus,
            );
        }
        Value::from_scalar(result_entries[0], st)
    } else {
        let mut result_shape = result_type.get_shape();
        // Insert 1 dims for the rank-1 cases, to simplify the broadcasting logic.
        if shape0.len() == 1 {
            shape0.insert(0, 1);
            result_shape.insert(result_shape.len() - 1, 1);
        }
        if shape1.len() == 1 {
            shape1.insert(1, 1);
            result_shape.insert(result_shape.len(), 1);
        }
        let middle_dim = shape1[shape1.len() - 2];
        for i in 0..(result_entries.len() as u64) {
            let result_index = number_to_index(i, &result_shape);
            for j in 0..middle_dim {
                let mut index0 = result_index
                    [result_shape.len() - shape0.len()..result_shape.len() - 1]
                    .to_vec();
                index0.push(j);
                let mut index1: Vec<u64>;
                index1 =
                    result_index[result_shape.len() - shape1.len()..result_shape.len()].to_vec();
                index1[shape1.len() - 2] = j;
                let n0 = index_to_number(&index0, &shape0);
                let n1 = index_to_number(&index1, &shape1);
                result_entries[i as usize] = add_u64(
                    result_entries[i as usize],
                    multiply_u64(entries0[n0 as usize], entries1[n1 as usize], modulus),
                    modulus,
                );
            }
        }
        Value::from_flattened_array(&result_entries, st)
    }
}

// This function can be heavily optimized, especially for binary input
fn evaluate_permute_axes(
    t: Type,
    value: Value,
    perm: ArrayShape,
    output_shape: ArrayShape,
) -> Result<Value> {
    let values = value.to_flattened_array_u64(t.clone())?;
    let cur_shape = t.get_shape();
    let mut result = vec![0u64; values.len()];
    for i in 0..values.len() as u64 {
        let old_index = number_to_index(i, &cur_shape);
        let mut new_index = vec![];
        for j in perm.iter() {
            new_index.push(old_index[*j as usize]);
        }
        result[index_to_number(&new_index, &output_shape) as usize] = values[i as usize];
    }
    Value::from_flattened_array(&result, t.get_scalar_type())
}

fn transpose_permutation(shape_length: usize) -> ArrayShape {
    let mut perm: Vec<u64> = (0..shape_length as u64).collect();
    if shape_length == 1 {
        return perm;
    }
    perm.swap(shape_length - 1, shape_length - 2);
    perm
}

fn evaluate_transpose_array(t: Type, value: Value) -> Result<Value> {
    let output_shape = transpose_shape(t.get_shape(), true);

    let perm = transpose_permutation(output_shape.len());

    // General case via permute axes
    // TODO: more efficient evaluation for bits
    evaluate_permute_axes(t, value, perm, output_shape)
}

fn general_gemm(
    trans_value0: Value,
    trans_value1: Value,
    trans_t0: Type,
    trans_t1: Type,
    result_type: Type,
) -> Result<Value> {
    let entries0 = trans_value0.to_flattened_array_u64(trans_t0.clone())?;
    let entries1 = trans_value1.to_flattened_array_u64(trans_t1.clone())?;

    let shape0 = trans_t0.get_shape();
    let shape1 = trans_t1.get_shape();

    let row_size = shape1[shape1.len() - 1] as usize;

    let st = trans_t0.get_scalar_type();
    let modulus = st.get_modulus();

    let result_length = {
        let result_shape = result_type.get_shape();
        result_shape.into_iter().product::<u64>() as usize
    };

    let mut result_entries = vec![0; result_length];
    let result_shape = result_type.get_shape();

    let n0 = shape0[shape0.len() - 2] as usize;
    let n1 = shape1[shape1.len() - 2] as usize;
    let result_matrix_size = n0 * n1;

    for matrix_i in (0..result_length).step_by(result_matrix_size) {
        // index of the first element in the current matrix, i.e. it ends with [...,0,0]
        let result_matrix_start_index = number_to_index(matrix_i as u64, &result_shape);

        let index0 = result_matrix_start_index
            [result_shape.len() - shape0.len()..result_shape.len()]
            .to_vec();
        let index1 = result_matrix_start_index
            [result_shape.len() - shape1.len()..result_shape.len()]
            .to_vec();

        let matrix_start_index0 = index_to_number(&index0, &shape0) as usize;
        let matrix_start_index1 = index_to_number(&index1, &shape1) as usize;
        for i in 0..n0 {
            let row0 = &entries0
                [matrix_start_index0 + i * row_size..matrix_start_index0 + (i + 1) * row_size];
            for j in 0..n1 {
                let row1 = &entries1
                    [matrix_start_index1 + j * row_size..matrix_start_index1 + (j + 1) * row_size];
                result_entries[matrix_i + i * n1 + j] = dot_vectors_u64(row0, row1, modulus)?;
            }
        }
    }
    Value::from_flattened_array(&result_entries, st)
}

fn evaluate_gemm(
    type0: Type,
    value0: Value,
    transpose0: bool,
    type1: Type,
    value1: Value,
    transpose1: bool,
    result_type: Type,
) -> Result<Value> {
    // Transpose both arrays such that the einsum operator ...ik, ...jk -> ...ij can be performed on them.
    // It means that the second array should be transposed if it is given in the correct form for matrix multiplication, i.e. it has shape ...kj.
    let trans_value0 = if transpose0 {
        evaluate_transpose_array(type0.clone(), value0)?
    } else {
        value0
    };
    let trans_value1 = if !transpose1 {
        evaluate_transpose_array(type1.clone(), value1)?
    } else {
        value1
    };

    let st = result_type.get_scalar_type();

    // Transpose input shapes
    let shape0 = transpose_shape(type0.get_shape(), transpose0);
    let shape1 = transpose_shape(type1.get_shape(), !transpose1);

    // Transposed types
    let trans_t0 = array_type(shape0, st.clone());
    let trans_t1 = array_type(shape1, st);

    general_gemm(trans_value0, trans_value1, trans_t0, trans_t1, result_type)
}

// Dummy value in Cuckoo hash tables that contain indices of arrays
const CUCKOO_DUMMY_ELEMENT: u64 = u64::MAX;

// Cuckoo hashing is computed as in <https://eprint.iacr.org/2018/579.pdf>, Section 3.2
fn evaluate_cuckoo(
    input_type: Type,
    input_value: Value,
    hash_matrices_type: Type,
    hash_matrices_value: Value,
    result_type: Type,
) -> Result<Value> {
    if !input_type.is_array() || !hash_matrices_type.is_array() {
        panic!("Inconsistency with type checker");
    }
    let input_shape = input_type.get_shape();
    let hash_matrices_shape = hash_matrices_type.get_shape();
    let input_bits = input_value.to_flattened_array_u64(input_type)?;
    let hash_matrices_bits = hash_matrices_value.to_flattened_array_u64(hash_matrices_type)?;
    let result_shape = result_type.get_shape();

    let size_of_output_table = result_shape[result_shape.len() - 1] as usize;
    let result_length = result_shape.into_iter().product::<u64>() as usize;

    // Initialize the hash table and table of used hash functions per element with dummy indices.
    let mut hash_table = vec![CUCKOO_DUMMY_ELEMENT; result_length];
    let mut used_hash_functions = vec![usize::MAX; result_length];

    let hash_functions = hash_matrices_shape[0] as usize;
    let hash_matrix_rows = hash_matrices_shape[1] as usize;
    let hash_matrix_columns = hash_matrices_shape[2] as usize;
    let hash_matrix_size = hash_matrix_rows * hash_matrix_columns;

    let num_input_sets = input_shape
        .iter()
        .take(input_shape.len() - 2)
        .product::<u64>() as usize;
    let num_input_strings_per_set = input_shape[input_shape.len() - 2] as usize;
    let input_string_length = input_shape[input_shape.len() - 1] as usize;

    for set_i in 0..num_input_sets {
        for string_i in 0..num_input_strings_per_set {
            let mut current_string_index = string_i;
            let mut current_hash_function_index = 0;
            let mut reinsert_attempt = 0;

            let mut insertion_failed = true;
            // If the number of consecutive re-insertions exceeds the bound, the hashing fails.
            // 100 is an empirical bound taken from <https://eprint.iacr.org/2018/579.pdf>, Appendix B.
            while reinsert_attempt < 100 {
                let string_start = (set_i * num_input_strings_per_set + current_string_index)
                    * input_string_length;
                let input_string = &input_bits[string_start..string_start + input_string_length];

                // Compute the hash of the input string
                let mut new_index = 0;
                // TODO: this matrix-vector product can be optimized
                for row in 0..hash_matrix_rows {
                    let mut hash_index_bit = 0;
                    for (column, input_bit) in
                        input_string.iter().enumerate().take(hash_matrix_columns)
                    {
                        hash_index_bit ^= hash_matrices_bits[hash_matrix_size
                            * current_hash_function_index
                            + row * hash_matrix_columns
                            + column]
                            & input_bit;
                    }
                    new_index ^= hash_index_bit << row;
                }

                // Check that the hash table is empty at the hash index
                let result_index = set_i * size_of_output_table + new_index as usize;
                if hash_table[result_index] == CUCKOO_DUMMY_ELEMENT {
                    // If yes, insert the current index into the hash table
                    // and the current hash function into the table of used hash functions
                    hash_table[result_index] = current_string_index as u64;
                    used_hash_functions[result_index] = current_hash_function_index;
                    insertion_failed = false;
                    break;
                } else {
                    // If no, extract the index and the corresponding hash function from the occupied cells
                    let old_string_index = hash_table[result_index] as usize;
                    let old_hash_function_index = used_hash_functions[result_index];
                    hash_table[result_index] = current_string_index as u64;
                    used_hash_functions[result_index] = current_hash_function_index;

                    // Re-insert the string with the extracted index using the next hash function
                    // NOTE: we change hash functions iteratively in contrast to the default random walk regime.
                    // It shouldn't significantly affect the failure probability as discussed in <https://eprint.iacr.org/2018/579.pdf>, Appendix B.
                    current_string_index = old_string_index;
                    current_hash_function_index = (old_hash_function_index + 1) % hash_functions;
                    reinsert_attempt += 1;
                }
            }
            if insertion_failed {
                return Err(runtime_error!("Cuckoo hashing failed"));
            }
        }
    }

    Value::from_flattened_array(&hash_table, UINT64)
}

// Fisher-Yates shuffle (<https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle>)
pub(crate) fn shuffle_array(array: &mut Vec<u64>, prng: &mut PRNG) -> Result<()> {
    for i in (1..array.len() as u64).rev() {
        let j = prng.get_random_in_range(Some(i + 1))?;
        array.swap(j as usize, i as usize);
    }
    Ok(())
}

fn evaluate_sum(node: Node, input_value: Value, axes: ArrayShape) -> Result<Value> {
    let dependency = node.get_node_dependencies()[0].clone();
    let inp_t = dependency.get_type()?;
    let values = input_value.to_flattened_array_u64(inp_t.clone())?;
    let res_t = node.get_type()?;
    match res_t {
        Type::Scalar(st) => {
            let mut result = 0u64;
            for v in values {
                result = add_u64(result, v, st.get_modulus());
            }
            Value::from_scalar(result, st)
        }
        Type::Array(res_shape, st) => {
            if axes.is_empty() {
                Ok(input_value)
            } else {
                let inp_shape = inp_t.get_shape();
                let res_len: u64 = res_shape.iter().product();
                let mut result = vec![0; res_len as usize];
                let mut res_axes = vec![];
                for j in 0..inp_shape.len() {
                    if !axes.contains(&(j as u64)) {
                        res_axes.push(j);
                    }
                }

                for (i, value) in values.iter().enumerate() {
                    let inp_index = number_to_index(i as u64, &inp_shape);
                    let mut new_index = vec![];
                    for ax in &res_axes {
                        new_index.push(inp_index[*ax]);
                    }
                    let new_i = index_to_number(&new_index, &res_shape) as usize;
                    result[new_i] = add_u64(result[new_i], *value, st.get_modulus());
                }
                Value::from_flattened_array(&result, st)
            }
        }
        _ => {
            panic!("Inconsistency between process_node() and evaluate()");
        }
    }
}

fn evaluate_cum_sum(node: Node, input_value: Value, axis: u64) -> Result<Value> {
    let t = node.get_node_dependencies()[0].get_type()?;
    let in_vec = input_value.to_flattened_array_u64(t.clone())?;
    let (shape, st) = match t {
        Type::Array(shape, st) => (shape, st),
        _ => return Err(runtime_error!("Inconsistency with type checker")),
    };
    let mut out_vec = in_vec.clone();
    for i in 0..in_vec.len() {
        let mut index = number_to_index(i as u64, &shape);
        if index[axis as usize] > 0 {
            index[axis as usize] -= 1;
            let j = index_to_number(&index, &shape) as usize;
            out_vec[i] = add_u64(out_vec[i], out_vec[j], st.get_modulus());
        }
    }
    Value::from_flattened_array(&out_vec, st)
}

// Choose `a` if `c = 1` and `b` if `c=0` in constant time.
//
// `c` must be equal to `0` or `1`.
//
// **WARNING**: This approach might have potential problems when compiled to WASM,
// see <https://blog.trailofbits.com/2022/01/26/part-1-the-life-of-an-optimization-barrier/>
#[inline(never)]
fn constant_time_select(a: u64, b: u64, c: u64) -> u64 {
    // Tells the compiler that the memory at &c is volatile and that it cannot make any assumptions about it.
    let mut c_per_bit = unsafe { core::ptr::read_volatile(&c as *const u64) };
    c_per_bit *= u64::MAX;
    c_per_bit & (a ^ b) ^ b
}

// TODO: consider pre-broadcasting constants where it is possible.
pub struct SimpleEvaluator {
    prng: PRNG,
    prfs: HashMap<Vec<u8>, Prf>,
}

impl SimpleEvaluator {
    pub fn new(prng_seed: Option<[u8; SEED_SIZE]>) -> Result<Self> {
        Ok(SimpleEvaluator {
            prng: PRNG::new(prng_seed)?,
            prfs: HashMap::new(),
        })
    }
}

impl Evaluator for SimpleEvaluator {
    fn evaluate_node(&mut self, node: Node, dependencies_values: Vec<Value>) -> Result<Value> {
        match node.get_operation() {
            Operation::Input(_) | Operation::Call | Operation::Iterate => {
                panic!("Should not be here!");
            }
            Operation::Zeros(t) => Ok(Value::zero_of_type(t)),
            Operation::Ones(t) => Ok(Value::one_of_type(t)?),
            Operation::Add
            | Operation::Subtract
            | Operation::Multiply
            | Operation::MixedMultiply => {
                let dependencies = node.get_node_dependencies();
                let value0_rc = dependencies_values[0].clone();
                let value1_rc = dependencies_values[1].clone();
                let type0 = dependencies[0].get_type()?;
                let type1 = dependencies[1].get_type()?;
                let result_value = if node.get_operation() == Operation::MixedMultiply {
                    evaluate_mixed_multiply(type0, value0_rc, type1, value1_rc, node.get_type()?)?
                } else {
                    evaluate_add_subtract_multiply(
                        type0,
                        value0_rc,
                        type1,
                        value1_rc,
                        node.get_operation(),
                        node.get_type()?,
                    )?
                };
                Ok(result_value)
            }
            Operation::Join(join_t, headers) => {
                let dependencies = node.get_node_dependencies();
                let res_t = node.get_type()?;

                let set0 = TypedValue {
                    value: dependencies_values[0].clone(),
                    t: dependencies[0].get_type()?,
                    name: None,
                };
                let set1 = TypedValue {
                    value: dependencies_values[1].clone(),
                    t: dependencies[1].get_type()?,
                    name: None,
                };

                evaluate_join(join_t, set0, set1, &headers, res_t)
            }
            Operation::CreateTuple
            | Operation::CreateNamedTuple(_)
            | Operation::CreateVector(_) => Ok(Value::from_vector(dependencies_values)),
            Operation::TupleGet(id) => Ok(dependencies_values[0].to_vector()?[id as usize].clone()),
            Operation::NamedTupleGet(ref field_name) => {
                let dependencies = node.get_node_dependencies();
                let tuple_type = dependencies[0].get_type()?;
                let mut field_id: Option<u64> = None;
                if let Type::NamedTuple(ref v) = tuple_type {
                    for (id, (current_field_name, _)) in v.iter().enumerate() {
                        if current_field_name.eq(field_name) {
                            field_id = Some(id as u64);
                            break;
                        }
                    }
                } else {
                    panic!("Inconsistency between type checker and evaluator");
                }
                let field_id_raw = field_id.unwrap();
                Ok(dependencies_values[0].to_vector()?[field_id_raw as usize].clone())
            }
            Operation::VectorGet => {
                let dependencies = node.get_node_dependencies();
                let index_type = dependencies[1].get_type()?;
                let index_value = dependencies_values[1].clone();
                let id = index_value.to_u64(index_type.get_scalar_type())?;
                let vector_type = dependencies[0].get_type()?;
                if let Type::Vector(size, _) = vector_type {
                    // id is unsigned, so it cannot be negative, we only need to check if it is not too big.
                    if id >= size {
                        return Err(runtime_error!("Index out of range"));
                    }
                } else {
                    panic!("Inconsistency with type checker.");
                }
                Ok(dependencies_values[0].to_vector()?[id as usize].clone())
            }
            Operation::Constant(_, value) => Ok(value),
            Operation::Zip => {
                let mut values = vec![];
                for value in dependencies_values {
                    values.push(value.to_vector()?);
                }
                let mut index = 0;
                let mut result = vec![];
                'result_entries: loop {
                    let mut row = vec![];
                    for value in &values {
                        if value.len() <= index {
                            break 'result_entries;
                        }
                        row.push(value[index].clone());
                    }
                    result.push(Value::from_vector(row));
                    index += 1;
                }
                Ok(Value::from_vector(result))
            }
            Operation::Stack(outer_shape) => {
                let dependencies = node.get_node_dependencies();
                let res_type = node.get_type()?;
                let full_shape = res_type.get_shape();
                let mut res_entries = vec![];
                let inner_shape = {
                    if full_shape == outer_shape {
                        vec![1]
                    } else {
                        full_shape[outer_shape.len()..].to_vec()
                    }
                };
                for i in 0..dependencies.len() {
                    let dep_type = dependencies[i].get_type()?;
                    let entries = match dep_type.clone() {
                        Type::Scalar(st) => {
                            vec![dependencies_values[i].to_u64(st)?]
                        }
                        Type::Array(_, _) => {
                            dependencies_values[i].to_flattened_array_u64(dep_type.clone())?
                        }
                        _ => {
                            panic!("Inconsistency with type checker.");
                        }
                    };
                    let mut resolved_entries =
                        broadcast_to_shape(&entries, &dep_type.get_dimensions(), &inner_shape);
                    res_entries.append(&mut resolved_entries);
                }
                Value::from_flattened_array(&res_entries, res_type.get_scalar_type())
            }
            Operation::A2B | Operation::B2A(_) | Operation::NOP => {
                Ok(dependencies_values[0].clone())
            }
            Operation::ArrayToVector => {
                let dependency = node.get_node_dependencies()[0].clone();
                let t = dependency.get_type()?;
                let values = dependencies_values[0].to_flattened_array_u64(t.clone())?;
                let shape = t.get_shape();
                let row_len: u64 = shape.iter().skip(1).product();
                let mut result = vec![];
                for row in values.chunks_exact(row_len as usize) {
                    result.push(Value::from_flattened_array(row, t.get_scalar_type())?);
                }
                Ok(Value::from_vector(result))
            }
            Operation::VectorToArray => {
                let values = dependencies_values[0].to_vector()?;
                let mut result = vec![];
                let t = node.get_type()?;
                let mut shape = t.get_shape();
                shape = shape[1..].to_vec();
                let st = t.get_scalar_type();
                if !shape.is_empty() {
                    for value in values {
                        let arr =
                            value.to_flattened_array_u64(array_type(shape.clone(), st.clone()))?;
                        result.extend_from_slice(&arr);
                    }
                } else {
                    for value in values {
                        let arr = value.to_u64(st.clone())?;
                        result.push(arr);
                    }
                }
                Value::from_flattened_array(&result, st)
            }
            Operation::Get(sub_index) => {
                let dependency = node.get_node_dependencies()[0].clone();
                let t = dependency.get_type()?;
                let values = dependencies_values[0].to_flattened_array_u64(t.clone())?;
                let shape = t.get_shape();
                let res_shape = shape[sub_index.len()..].to_vec();
                let res_len: u64 = res_shape.iter().product();
                let sub_index_num = index_to_number(&sub_index, &shape[..sub_index.len()]);
                let result = values
                    .chunks_exact(res_len as usize)
                    .nth(sub_index_num as usize)
                    .unwrap();
                Value::from_flattened_array(result, t.get_scalar_type())
            }
            Operation::GetSlice(slice) => {
                let dependency = node.get_node_dependencies()[0].clone();
                let dependency_type = dependency.get_type()?;
                let dependency_value =
                    dependencies_values[0].to_flattened_array_u64(dependency_type.clone())?;
                let dependency_shape = dependency_type.get_shape();
                let result_type = node.get_type()?;
                let result_shape = result_type.get_dimensions();
                let mut result = vec![];
                for i in 0..result_shape.iter().product() {
                    let index = number_to_index(i, &result_shape);
                    let dependency_index =
                        slice_index(dependency_shape.clone(), slice.clone(), index.clone())?;
                    let j = index_to_number(&dependency_index, &dependency_shape);
                    result.push(dependency_value[j as usize]);
                }
                Value::from_flattened_array(&result, result_type.get_scalar_type())
            }
            Operation::PermuteAxes(perm) => {
                let dependency = node.get_node_dependencies()[0].clone();
                let t = dependency.get_type()?;
                let res_shape = node.get_type()?.get_shape();

                evaluate_permute_axes(t, dependencies_values[0].clone(), perm, res_shape)
            }
            Operation::InversePermutation => {
                let dependency = node.get_node_dependencies()[0].clone();
                let t = dependency.get_type()?;
                let values = dependencies_values[0].to_flattened_array_u64(t.clone())?;
                let mut values_without_dup = values.clone();
                values_without_dup.sort_unstable();
                values_without_dup.dedup();
                if values.len() != values_without_dup.len() {
                    return Err(runtime_error!(
                        "Input array doesn't contain a valid permutation"
                    ));
                }
                let result = execute_inverse_permutation(values)?;
                Value::from_flattened_array(&result, t.get_scalar_type())
            }
            Operation::Sum(axes) => evaluate_sum(node, dependencies_values[0].clone(), axes),
            Operation::CumSum(axis) => evaluate_cum_sum(node, dependencies_values[0].clone(), axis),
            Operation::Reshape(new_type) => {
                let dependency_value = dependencies_values[0].clone();
                let dependency_value_flattened = flatten_value(dependency_value);
                let new_value = unflatten_value(&dependency_value_flattened, &mut 0, new_type);
                Ok(new_value)
            }
            Operation::ApplyPermutation(inverse_permutation) => {
                let t = node.get_type()?;
                let n = t.get_shape()[0];

                let indexes_permutation = dependencies_values[1]
                    .to_flattened_array_u64(node.get_node_dependencies()[1].get_type()?)?;
                if indexes_permutation
                    .iter()
                    .cloned()
                    .filter(|&x| x < n)
                    .collect::<HashSet<u64>>()
                    .len()
                    != n as usize
                {
                    return Err(runtime_error!(
                        "Argument 1 doesn't contain a valid permutation."
                    ));
                }
                let permutation = if inverse_permutation {
                    execute_inverse_permutation(indexes_permutation)?
                } else {
                    indexes_permutation
                };
                evaluate_gather(
                    TypedValue::new(t.clone(), dependencies_values[0].clone())?,
                    permutation,
                    t,
                    0,
                )
            }
            Operation::Sort(key) => {
                let tv = TypedValue::new(
                    node.get_node_dependencies()[0].get_type()?,
                    dependencies_values[0].clone(),
                )?;
                let arrays = tv.to_vector()?;
                let mut key_array = None;
                let key = Some(key);
                for tv in arrays.iter() {
                    if tv.name == key {
                        key_array = Some(tv.clone());
                        break;
                    }
                }
                let key_array = key_array.ok_or_else(|| {
                    runtime_error!("Input doesn't contain a key named {:?}.", key)
                })?;
                let t = key_array.t.clone();
                let n = t.get_shape()[0];
                let sorting_permutation =
                    get_sorting_permutation(key_array.value.to_flattened_array_u64(t)?, n)?;

                let mut result = vec![];
                for array in arrays {
                    result.push(evaluate_gather(
                        array.clone(),
                        sorting_permutation.clone(),
                        array.t,
                        0,
                    )?);
                }

                Ok(Value::from_vector(result))
            }
            Operation::Truncate(scale) => {
                // For signed scalar type, we interpret a number 0 <= x < modulus as follows:
                // If x < modulus / 2, then it is treated as x, otherwise,
                // as x - modulus.
                let dependency = node.get_node_dependencies()[0].clone();
                let dependency_type = dependency.get_type()?;
                let scalar_type = dependency_type.get_scalar_type();
                let dependency_value = dependencies_values[0].clone();
                let mut entries = if dependency_type.is_scalar() {
                    vec![dependency_value.to_u64(scalar_type.clone())?]
                } else {
                    dependency_value.to_flattened_array_u64(dependency_type.clone())?
                };
                for entry in &mut entries {
                    if scalar_type.get_signed() {
                        match scalar_type.get_modulus() {
                            Some(modulus) => {
                                let mut val = *entry as i64;
                                if val >= (modulus / 2) as i64 {
                                    val -= modulus as i64;
                                }
                                let mut res = val / (scale as i64);
                                if res < 0 {
                                    res += modulus as i64;
                                }
                                *entry = res as u64;
                            }
                            None => {
                                *entry = ((*entry as i64) / (scale as i64)) as u64;
                            }
                        }
                    } else {
                        *entry /= scale;
                    }
                }
                let new_value = if dependency_type.is_scalar() {
                    Value::from_scalar(entries[0], dependency_type.get_scalar_type())?
                } else {
                    Value::from_flattened_array(&entries, dependency_type.get_scalar_type())?
                };
                Ok(new_value)
            }
            Operation::Repeat(n) => {
                let dependency_value = dependencies_values[0].clone();
                let v: Vec<Value> = repeat(dependency_value).take(n as usize).collect();
                let new_value = Value::from_vector(v);
                Ok(new_value)
            }
            Operation::Dot => {
                let dependency0 = node.get_node_dependencies()[0].clone();
                let type0 = dependency0.get_type()?;
                let value0 = dependencies_values[0].clone();
                let dependency1 = node.get_node_dependencies()[1].clone();
                let type1 = dependency1.get_type()?;
                let value1 = dependencies_values[1].clone();
                let result_type = node.get_type()?;
                let result_value = evaluate_dot(type0, value0, type1, value1, result_type)?;
                Ok(result_value)
            }
            Operation::Matmul => {
                let dependency0 = node.get_node_dependencies()[0].clone();
                let type0 = dependency0.get_type()?;
                let value0 = dependencies_values[0].clone();
                let dependency1 = node.get_node_dependencies()[1].clone();
                let type1 = dependency1.get_type()?;
                let value1 = dependencies_values[1].clone();
                let result_type = node.get_type()?;
                let result_value = evaluate_matmul(type0, value0, type1, value1, result_type)?;
                Ok(result_value)
            }
            Operation::Gemm(transpose0, transpose1) => {
                let dependency0 = node.get_node_dependencies()[0].clone();
                let type0 = dependency0.get_type()?;
                let value0 = dependencies_values[0].clone();
                let dependency1 = node.get_node_dependencies()[1].clone();
                let type1 = dependency1.get_type()?;
                let value1 = dependencies_values[1].clone();
                let result_type = node.get_type()?;
                evaluate_gemm(
                    type0,
                    value0,
                    transpose0,
                    type1,
                    value1,
                    transpose1,
                    result_type,
                )
            }
            Operation::Random(t) => {
                let new_value = self.prng.get_random_value(t)?;
                Ok(new_value)
            }
            Operation::RandomPermutation(n) => {
                let mut result_array: Vec<u64> = (0..n).collect();

                shuffle_array(&mut result_array, &mut self.prng)?;

                Value::from_flattened_array(&result_array, UINT64)
            }
            Operation::DecomposeSwitchingMap(n) => {
                let input_node = node.get_node_dependencies()[0].clone();
                let t = input_node.get_type()?;
                let input_value = dependencies_values[0].clone();
                let input_array = input_value.to_flattened_array_u64(t.clone())?;

                let input_shape = t.get_shape();
                let num_maps = input_shape
                    .iter()
                    .take(input_shape.len() - 1)
                    .product::<u64>() as usize;
                let map_size = input_shape[input_shape.len() - 1] as usize;
                // Permutation with deletion map
                let mut perm1_array = vec![];
                // Duplication map
                let mut duplication_map = vec![];
                // Duplication bits
                let mut duplication_bits = vec![];
                // Permutation without deletion map
                let mut perm2_array = vec![];

                for map_i in 0..num_maps {
                    let map_start = map_i * map_size;

                    // Permutation with deletion
                    let mut little_perm1_array = vec![];
                    // Permutation used for grouping identical indices of the input switching map
                    let mut perm_from_switch_to_perm1 = vec![];
                    // Duplication map
                    let mut little_duplication_map: Vec<u64> = vec![];
                    // Duplication bits
                    let mut little_duplication_bits = vec![];

                    // true if index isn't present in the map
                    let mut missing_indices_flags = vec![true; n as usize];
                    let mut existing_indices = vec![];

                    // Hash map with the locations of the switching map elements
                    let mut switch_indexes: HashMap<u64, Vec<u64>> = HashMap::new();
                    for i in 0..map_size {
                        let input_index = input_array[map_start + i];
                        if input_index >= n {
                            return Err(runtime_error!("Switching map has incorrect indices"));
                        }
                        if let Some(v) = switch_indexes.get_mut(&input_index) {
                            v.push(i as u64);
                        } else {
                            switch_indexes.insert(input_index, vec![i as u64]);
                            existing_indices.push(input_index);
                        }
                        missing_indices_flags[input_index as usize] = false;
                    }

                    // Indices not present in the switching map
                    let mut missing_indices = vec![];
                    for (i, flag) in missing_indices_flags.iter().enumerate() {
                        if *flag {
                            missing_indices.push(i as u64);
                        }
                    }
                    // Randomize the order of remaining indices
                    shuffle_array(&mut missing_indices, &mut self.prng)?;

                    // Indices that didn't appear in the switching map
                    let mut missing_indices_index = 0;

                    for input_index in existing_indices {
                        let locations_vec = switch_indexes.get(&input_index).unwrap();
                        let num_copies = locations_vec.len();
                        little_perm1_array.push(input_index);
                        let current_dup_index = little_perm1_array.len() as u64 - 1;
                        little_duplication_map.push(current_dup_index);
                        little_duplication_bits.push(0u64);
                        for _ in 0..num_copies - 1 {
                            little_perm1_array.push(missing_indices[missing_indices_index]);
                            little_duplication_map.push(current_dup_index);
                            little_duplication_bits.push(1);
                            missing_indices_index += 1;
                        }
                        perm_from_switch_to_perm1.extend_from_slice(locations_vec);
                    }

                    // Invert permutation that was used for grouping identical indices of the input switching map
                    let mut little_perm2_array = vec![0; map_size];
                    for i in 0..map_size {
                        little_perm2_array[perm_from_switch_to_perm1[i] as usize] = i;
                    }

                    perm1_array.extend_from_slice(&little_perm1_array);
                    duplication_map.extend_from_slice(&little_duplication_map);
                    duplication_bits.extend_from_slice(&little_duplication_bits);
                    perm2_array.extend_from_slice(&little_perm2_array);
                }

                let perm1_val = Value::from_flattened_array(&perm1_array, UINT64)?;
                let dup_map_val = Value::from_flattened_array(&duplication_map, UINT64)?;
                let dup_bits_val = Value::from_flattened_array(&duplication_bits, BIT)?;
                let perm2_val = Value::from_flattened_array(&perm2_array, UINT64)?;
                Ok(Value::from_vector(vec![
                    perm1_val,
                    Value::from_vector(vec![dup_map_val, dup_bits_val]),
                    perm2_val,
                ]))
            }
            Operation::CuckooToPermutation => {
                let input_node = node.get_node_dependencies()[0].clone();
                let t = input_node.get_type()?;
                let input_value = dependencies_values[0].clone();
                let input_array = input_value.to_flattened_array_u64(t.clone())?;

                let input_shape = t.get_shape();
                let num_cuckoo_tables = input_shape
                    .iter()
                    .take(input_shape.len() - 1)
                    .product::<u64>();
                let table_size = input_shape[input_shape.len() - 1];
                let mut result_array = vec![0; (num_cuckoo_tables * table_size) as usize];

                for table_i in 0..num_cuckoo_tables as usize {
                    let mut num_dummies = 0;
                    let table_start = table_i * table_size as usize;
                    for i in 0..table_size as usize {
                        // Compute the bit input element == CUCKOO_DUMMY_ELEMENT using the fact that CUCKOO_DUMMY_ELEMENT = u64::MAX
                        num_dummies += input_array[table_start + i] / CUCKOO_DUMMY_ELEMENT;
                    }
                    // Check that after removing the dummies there are no other duplicates removed
                    let mut input_wout_dup =
                        input_array[table_start..table_start + table_size as usize].to_vec();
                    input_wout_dup.sort_unstable();
                    input_wout_dup.dedup();
                    if num_dummies > 1 {
                        if input_wout_dup.len() as u64 + num_dummies - 1 != table_size {
                            return Err(runtime_error!("Input array contains duplicate indices"));
                        }
                    } else if input_wout_dup.len() as u64 != table_size {
                        return Err(runtime_error!("Input array contains duplicate indices"));
                    }
                    let mut remaining_indices: Vec<u64> =
                        (table_size - num_dummies..table_size).collect();
                    // If there are no dummy elements, set remaining indices to [CUCKOO_DUMMY_ELEMENT] to support the constant-time selection below.
                    if remaining_indices.is_empty() {
                        remaining_indices.push(CUCKOO_DUMMY_ELEMENT);
                    }
                    // Shuffle remaining indices
                    shuffle_array(&mut remaining_indices, &mut self.prng)?;
                    let mut current_index = 0;
                    for i in 0..table_size as usize {
                        // Check that non-dummy elements of the Cuckoo table are correct indices of an array of length `table_size - num_dummies`.
                        if input_array[table_start + i] >= table_size - num_dummies
                            && input_array[table_start + i] != CUCKOO_DUMMY_ELEMENT
                        {
                            return Err(runtime_error!("Indices are incorrect"));
                        }
                        // Compute the bit input element == CUCKOO_DUMMY_ELEMENT using the fact that CUCKOO_DUMMY_ELEMENT = u64::MAX
                        let is_dummy = input_array[table_start + i] / CUCKOO_DUMMY_ELEMENT;
                        // Select either an input array element or a random index if this element is dummy
                        // Select in constant time to avoid possible leakage of dummy positions
                        result_array[table_start + i] = constant_time_select(
                            remaining_indices[current_index],
                            input_array[table_start + i],
                            is_dummy,
                        );
                        current_index = min(
                            current_index + is_dummy as usize,
                            remaining_indices.len() - 1,
                        );
                    }
                }
                Value::from_flattened_array(&result_array, UINT64)
            }
            Operation::PRF(iv, t) => {
                let key_value = dependencies_values[0].clone();
                let key = key_value.access_bytes(|bytes| Ok(bytes.to_vec()))?;
                // at this point the PRF map should be of the Some type
                let new_value = match self.prfs.entry(key.clone()) {
                    Entry::Vacant(e) => {
                        let mut key_slice = [0u8; SEED_SIZE];
                        key_slice.copy_from_slice(&key[0..SEED_SIZE]);
                        let mut prf = Prf::new(Some(key_slice))?;
                        let val = prf.output_value(iv, t)?;
                        e.insert(prf);
                        val
                    }
                    Entry::Occupied(mut e) => {
                        let prf = e.get_mut();
                        prf.output_value(iv, t)?
                    }
                };
                Ok(new_value)
            }
            Operation::PermutationFromPRF(iv, n) => {
                let key_value = dependencies_values[0].clone();
                let key = key_value.access_bytes(|bytes| Ok(bytes.to_vec()))?;
                // at this point the PRF map should be of the Some type
                let new_value = match self.prfs.entry(key.clone()) {
                    Entry::Vacant(e) => {
                        let mut key_slice = [0u8; SEED_SIZE];
                        key_slice.copy_from_slice(&key[0..SEED_SIZE]);
                        let mut prf = Prf::new(Some(key_slice))?;
                        let val = prf.output_permutation(iv, n)?;
                        e.insert(prf);
                        val
                    }
                    Entry::Occupied(mut e) => {
                        let prf = e.get_mut();
                        prf.output_permutation(iv, n)?
                    }
                };
                Ok(new_value)
            }
            Operation::CuckooHash => {
                let input_value = dependencies_values[0].clone();
                let hash_matrices_value = dependencies_values[1].clone();

                let input_type = node.get_node_dependencies()[0].get_type()?;
                let hash_matrices_type = node.get_node_dependencies()[1].get_type()?;

                let result_type = node.get_type()?;
                evaluate_cuckoo(
                    input_type,
                    input_value,
                    hash_matrices_type,
                    hash_matrices_value,
                    result_type,
                )
            }
            Operation::SegmentCumSum => {
                let input_array_value = dependencies_values[0].clone();
                let binary_array_value = dependencies_values[1].clone();
                let first_row_value = dependencies_values[2].clone();

                let input_t = node.get_node_dependencies()[0].get_type()?;
                let binary_t = node.get_node_dependencies()[1].get_type()?;
                let first_row_t = node.get_node_dependencies()[2].get_type()?;

                let input_array = input_array_value.to_flattened_array_u64(input_t.clone())?;
                let binary_array = binary_array_value.to_flattened_array_u64(binary_t)?;
                let input_st = input_t.get_scalar_type();
                let first_row = if first_row_t.is_scalar() {
                    vec![first_row_value.to_u64(input_st.clone())?]
                } else {
                    first_row_value.to_flattened_array_u64(first_row_t.clone())?
                };

                let row_size = first_row_t.get_dimensions().iter().product::<u64>() as usize;

                let mut result_array = first_row;
                for (i, b) in binary_array.iter().enumerate() {
                    let mut result_row = if *b == 0 {
                        input_array[i * row_size..(i + 1) * row_size].to_vec()
                    } else {
                        // Extract an input row and sum it with the previous output row
                        let previous_row = &result_array[i * row_size..(i + 1) * row_size];
                        let input_row = &input_array[i * row_size..(i + 1) * row_size];
                        add_vectors_u64(input_row, previous_row, input_st.get_modulus())?
                    };
                    result_array.append(&mut result_row);
                }

                Value::from_flattened_array(&result_array, input_st)
            }
            Operation::Gather(axis) => {
                let input = TypedValue::new(
                    node.get_node_dependencies()[0].get_type()?,
                    dependencies_values[0].clone(),
                )?;
                let indices = dependencies_values[1]
                    .to_flattened_array_u64(node.get_node_dependencies()[1].get_type()?)?;
                evaluate_gather(input, indices, node.get_type()?, axis)
            }
            Operation::Concatenate(axis) => {
                let dependencies = node.get_node_dependencies();
                let result_t = node.get_type()?;
                let result_shape = result_t.get_shape();

                // number of arrays within which sub-arrays are concatenated
                let num_arrays = result_shape.iter().take(axis as usize).product::<u64>();
                // element size of concatenated arrays
                let item_length = result_shape.iter().skip(axis as usize + 1).product::<u64>();

                let mut dependencies_arrays = vec![];
                // number of elements to concatenate of every dependency
                let mut dependencies_num_items = vec![];
                for (i, value) in dependencies_values.iter().enumerate() {
                    let t = dependencies[i].get_type()?;
                    dependencies_arrays.push(value.to_flattened_array_u64(t.clone())?);
                    dependencies_num_items.push(t.get_shape()[axis as usize]);
                }

                let mut result_array: Vec<u64> = vec![];
                for array_i in 0..num_arrays {
                    for (dep_i, dep_array) in dependencies_arrays.iter().enumerate() {
                        let num_items = dependencies_num_items[dep_i];
                        let start = array_i * num_items * item_length;
                        result_array.extend(
                            &dep_array[start as usize..(start + num_items * item_length) as usize],
                        );
                    }
                }

                Value::from_flattened_array(&result_array, result_t.get_scalar_type())
            }
            Operation::Print(message) => {
                if dependencies_values.len() != 1 {
                    return Err(runtime_error!(
                        "Inconsistency with type checker, Print should have 1 dependency, got {}",
                        dependencies_values.len()
                    ));
                }
                let t = node.get_node_dependencies()[0].get_type()?;
                let val = dependencies_values[0].clone();
                let tv = TypedValue::new(t, val.clone())?;
                eprintln!("{message}: {tv:?}");
                Ok(val)
            }
            Operation::Assert(message) => {
                if dependencies_values.len() != 2 {
                    return Err(runtime_error!("Inconsistency with type checker, Assert should have 2 dependencies, got {}", dependencies_values.len()));
                }
                let bit_val = dependencies_values[0].clone();
                if !bit_val.to_bit()? {
                    return Err(runtime_error!("Assertion failed: {message}"));
                }
                Ok(dependencies_values[1].clone())
            }
            _ => Err(runtime_error!("Not implemented")),
        }
    }
}

fn get_sorting_permutation<T: Clone + Eq + PartialEq + Ord + PartialOrd>(
    array: Vec<T>,
    n: u64,
) -> Result<Vec<u64>> {
    let chunk_size = array.len() / n as usize;
    let mut enumerated = array
        .chunks(chunk_size)
        .map(|x| x.to_vec())
        .zip(0..n)
        .collect::<Vec<(Vec<T>, u64)>>();
    // Stable sort.
    enumerated.sort_by(|a, b| a.0.cmp(&b.0));
    let indexes_permutation = enumerated.into_iter().map(|x| x.1).collect::<Vec<u64>>();
    Ok(indexes_permutation)
}

fn execute_inverse_permutation(values: Vec<u64>) -> Result<Vec<u64>> {
    let mut result = vec![0u64; values.len()];
    for i in 0..values.len() {
        let value = values[i] as usize;
        if value >= values.len() {
            return Err(runtime_error!(
                "Input array doesn't contain a valid permutation"
            ));
        }
        result[value] = i as u64;
    }
    Ok(result)
}

fn evaluate_gather(
    input: TypedValue,
    indices: Vec<u64>,
    result_type: Type,
    axis: u64,
) -> Result<Value> {
    let input_entries = input.value.to_flattened_array_u64(input.t.clone())?;

    let mut output_entries = vec![];

    let input_shape = input.t.get_shape();

    // Number of subarrays whose indices are selected
    let num_arrays = input_shape[..axis as usize]
        .to_vec()
        .iter()
        .product::<u64>();

    // Number of elements in each row indexed by the indices
    let row_size = input_shape[(axis + 1) as usize..]
        .to_vec()
        .iter()
        .product::<u64>();

    for array_i in 0..num_arrays {
        for index_entry in indices.iter() {
            if *index_entry >= input_shape[axis as usize] {
                return Err(runtime_error!("Incorrect index"));
            }
            let input_flat_index = (array_i * input_shape[axis as usize] + index_entry) * row_size;
            output_entries.extend_from_slice(
                &input_entries[input_flat_index as usize..(input_flat_index + row_size) as usize],
            );
        }
    }
    Value::from_flattened_array(&output_entries, result_type.get_scalar_type())
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::{
        data_types::{
            named_tuple_type, scalar_type, tuple_type, vector_type, ArrayShape, ScalarType, INT16,
            INT32, INT64, UINT16, UINT32, UINT64, UINT8,
        },
        evaluators::{evaluate_simple_evaluator, random_evaluate},
        graphs::{create_context, util::simple_context, JoinType, SliceElement},
        random::chi_statistics,
        type_inference::NULL_HEADER,
        typed_value_operations::{FromVectorMode, TypedValueArrayOperations, TypedValueOperations},
    };

    use super::*;

    #[test]
    fn test_prf() {
        let helper = |iv: u64, t: Type| -> Result<()> {
            let c = simple_context(|g| {
                let i1 = g.random(array_type(vec![128], BIT))?;
                let i2 = g.random(array_type(vec![128], BIT))?;
                let p1 = g.prf(i1.clone(), iv, t.clone())?;
                let p2 = g.prf(i2, iv, t.clone())?;
                let p3 = g.prf(i1, iv, t.clone())?;
                g.create_vector(t.clone(), vec![p1, p2, p3])
            })?;
            let mut evaluator = SimpleEvaluator {
                prng: PRNG::new(None)?,
                prfs: HashMap::new(),
            };
            let v = evaluator.evaluate_context(c, Vec::new())?;
            let ot = vector_type(3, t.clone());
            assert_eq!(evaluator.prfs.len(), 2);
            assert!(v.check_type(ot)?);
            Ok(())
        };
        || -> Result<()> {
            helper(0, scalar_type(BIT))?;
            helper(1, scalar_type(UINT8))?;
            helper(2, scalar_type(INT32))?;
            helper(3, array_type(vec![2, 5], BIT))?;
            helper(4, array_type(vec![2, 5], UINT8))?;
            helper(5, array_type(vec![2, 5], INT32))?;
            helper(6, tuple_type(vec![scalar_type(BIT), scalar_type(INT32)]))?;
            helper(
                7,
                tuple_type(vec![
                    vector_type(3, scalar_type(BIT)),
                    vector_type(5, scalar_type(BIT)),
                    scalar_type(BIT),
                    scalar_type(INT32),
                ]),
            )?;
            helper(
                8,
                named_tuple_type(vec![
                    ("field 1".to_owned(), scalar_type(BIT)),
                    ("field 2".to_owned(), scalar_type(INT32)),
                ]),
            )
        }()
        .unwrap()
    }

    fn cuckoo_helper(
        input_shape: ArrayShape,
        hash_shape: ArrayShape,
        inputs: Vec<Value>,
    ) -> Result<Vec<u64>> {
        let c = simple_context(|g| {
            let i = g.input(array_type(input_shape.clone(), BIT))?;
            let hash_matrix = g.input(array_type(hash_shape.clone(), BIT))?;
            i.cuckoo_hash(hash_matrix)
        })?;
        let g = c.get_main_graph()?;
        let o = g.get_output_node()?;
        let result_value = random_evaluate(g, inputs)?;
        let result_type = o.get_type()?;
        result_value.to_flattened_array_u64(result_type)
    }

    #[test]
    fn test_cuckoo_hash() {
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
                // output [4]-array
                // Hashing results in: h_0(input[0]) = 00, h_0(input[1]) = 10
                let expected = vec![0, 1, u64::MAX, u64::MAX];
                assert_eq!(
                    cuckoo_helper(vec![2, 3], vec![3, 2, 3], vec![input, hash_matrix])?,
                    expected
                );
            }
            // collision
            {
                // [2,3]-array
                let input = Value::from_flattened_array(&[1, 0, 1, 0, 0, 0], BIT)?;
                // [3,2,3]-array
                let hash_matrix = Value::from_flattened_array(
                    &[1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1],
                    BIT,
                )?;
                // output [4]-array
                // Hashing results in:
                // h_0(input[0]) = 00, h_0(input[1]) = 00
                // h_1(input[0]) = 00, h_1(input[1]) = 00
                // h_2(input[0]) = 11
                let expected = vec![1, u64::MAX, u64::MAX, 0];
                assert_eq!(
                    cuckoo_helper(vec![2, 3], vec![3, 2, 3], vec![input, hash_matrix])?,
                    expected
                );
            }
            // failure
            {
                // [2,3]-array
                let input = Value::from_flattened_array(&[1, 0, 1, 0, 0, 0], BIT)?;
                // [3,2,3]-array
                // Hashes everything to 0
                let hash_matrix = Value::from_flattened_array(&[0; 18], BIT)?;
                let e = cuckoo_helper(vec![2, 3], vec![3, 2, 3], vec![input, hash_matrix]);
                assert!(e.is_err());
            }
            // somewhat big example
            for _ in 0..1000 {
                let mut prng = PRNG::new(None)?;
                // Probability of getting two equal strings in each [512,32]-subarray is 3*10^(-5) by the birthday paradox
                let input_shape = vec![512, 32];
                let input = prng.get_random_value(array_type(input_shape.clone(), BIT))?;
                // The size of hash table per each input set is 1024.
                // Thus, the number of rows of hash matrices is log_2(1024) = 10.
                // The estimated failure probability is ~2^(-108) according to <https://eprint.iacr.org/2018/579.pdf>, Appendix B.
                // However, the probability that there is a pair of elements with hashes equal to a fixed value is Omega(1/1024^3).
                let hash_shape = vec![3, 10, 32];
                let hash_matrix = prng.get_random_value(array_type(hash_shape.clone(), BIT))?;
                assert!(cuckoo_helper(input_shape, hash_shape, vec![input, hash_matrix]).is_ok());
            }
            Ok(())
        }()
        .unwrap();
    }

    fn segment_cumsum_helper(
        input_shape: ArrayShape,
        st: ScalarType,
        inputs: Vec<Value>,
    ) -> Result<Vec<u64>> {
        let c = simple_context(|g| {
            let i = g.input(array_type(input_shape.clone(), st.clone()))?;
            let b = g.input(array_type(vec![input_shape[0]], BIT))?;
            let first_row = if input_shape.len() > 1 {
                g.input(array_type(input_shape[1..].to_vec(), st))?
            } else {
                g.input(scalar_type(st))?
            };
            i.segment_cumsum(b, first_row)
        })?;
        let g = c.get_main_graph()?;
        let o = g.get_output_node()?;
        let result_value = random_evaluate(g, inputs)?;
        let result_type = o.get_type()?;
        result_value.to_flattened_array_u64(result_type)
    }

    #[test]
    fn test_segment_cumsum() {
        || -> Result<()> {
            {
                let input = Value::from_flattened_array(&[1, 2, 3, 4, 5, 6], INT32)?;
                let binary = Value::from_flattened_array(&[0, 1, 1, 0, 0, 1], BIT)?;
                let first_row = Value::from_scalar(10, INT32)?;

                let expected = vec![10, 1, 3, 6, 4, 5, 11];
                assert_eq!(
                    segment_cumsum_helper(vec![6], INT32, vec![input, binary, first_row])?,
                    expected
                );
            }
            {
                let input = Value::from_flattened_array(&[1, 2, 3, 4, 5, 6], INT32)?;
                let binary = Value::from_flattened_array(&[0, 0, 0, 0, 0, 0], BIT)?;
                let first_row = Value::from_scalar(10, INT32)?;

                let expected = vec![10, 1, 2, 3, 4, 5, 6];
                assert_eq!(
                    segment_cumsum_helper(vec![6], INT32, vec![input, binary, first_row])?,
                    expected
                );
            }
            {
                let input = Value::from_flattened_array(&[1, 2, 3, 4, 5, 6], INT32)?;
                let binary = Value::from_flattened_array(&[1, 1, 1, 1, 1, 1], BIT)?;
                let first_row = Value::from_scalar(10, INT32)?;

                let expected = vec![10, 11, 13, 16, 20, 25, 31];
                assert_eq!(
                    segment_cumsum_helper(vec![6], INT32, vec![input, binary, first_row])?,
                    expected
                );
            }
            {
                let input =
                    Value::from_ndarray(array!([[1, 2], [3, 4], [5, 6]]).into_dyn(), INT32)?;
                let binary = Value::from_flattened_array(&[0, 1, 1], BIT)?;
                let first_row = Value::from_flattened_array(&[10, 20], INT32)?;

                let expected = array!([[10, 20], [1, 2], [4, 6], [9, 12]]).into_raw_vec();
                assert_eq!(
                    segment_cumsum_helper(vec![3, 2], INT32, vec![input, binary, first_row])?,
                    expected
                );
            }
            {
                let input =
                    Value::from_ndarray(array!([[1, 2], [3, 4], [5, 6]]).into_dyn(), INT32)?;
                let binary = Value::from_flattened_array(&[1, 1, 1], BIT)?;
                let first_row = Value::from_flattened_array(&[10, 20], INT32)?;

                let expected = array!([[10, 20], [11, 22], [14, 26], [19, 32]]).into_raw_vec();
                assert_eq!(
                    segment_cumsum_helper(vec![3, 2], INT32, vec![input, binary, first_row])?,
                    expected
                );
            }
            {
                let input =
                    Value::from_ndarray(array!([[1, 2], [3, 4], [5, 6]]).into_dyn(), INT32)?;
                let binary = Value::from_flattened_array(&[0, 0, 0], BIT)?;
                let first_row = Value::from_flattened_array(&[10, 20], INT32)?;

                let expected = array!([[10, 20], [1, 2], [3, 4], [5, 6]]).into_raw_vec();
                assert_eq!(
                    segment_cumsum_helper(vec![3, 2], INT32, vec![input, binary, first_row])?,
                    expected
                );
            }

            Ok(())
        }()
        .unwrap();
    }

    fn inverse_permutation_helper(n: u64, inputs: Vec<Value>) -> Result<Vec<u64>> {
        let input_type = array_type(vec![n], UINT64);
        let c = simple_context(|g| g.input(input_type.clone())?.inverse_permutation())?;
        let g = c.get_main_graph()?;
        let result_value = random_evaluate(g, inputs)?;
        result_value.to_flattened_array_u64(input_type)
    }

    fn gather_helper(
        input_shape: ArrayShape,
        indices_shape: ArrayShape,
        axis: u64,
        inputs: Vec<Value>,
    ) -> Result<Vec<u64>> {
        let c = simple_context(|g| {
            let inp = g.input(array_type(input_shape.clone(), UINT32))?;
            let ind = g.input(array_type(indices_shape.clone(), UINT64))?;
            inp.gather(ind, axis)
        })?;
        let g = c.get_main_graph()?;
        let o = g.get_output_node()?;
        let result_value = random_evaluate(g, inputs)?;
        let result_type = o.get_type()?;
        result_value.to_flattened_array_u64(result_type)
    }

    #[test]
    fn test_inverse_permutation() {
        || -> Result<()> {
            {
                let input = Value::from_flattened_array(&[0], UINT64)?;
                let expected = vec![0];
                assert_eq!(inverse_permutation_helper(1, vec![input])?, expected);
            }
            {
                let input = Value::from_flattened_array(&[0, 1, 2, 3, 4], UINT64)?;
                let expected = vec![0, 1, 2, 3, 4];
                assert_eq!(inverse_permutation_helper(5, vec![input])?, expected);
            }
            {
                let input = Value::from_flattened_array(&[4, 3, 2, 1, 0], UINT64)?;
                let expected = vec![4, 3, 2, 1, 0];
                assert_eq!(inverse_permutation_helper(5, vec![input])?, expected);
            }
            {
                let input = Value::from_flattened_array(&[2, 0, 1, 4, 3], UINT64)?;
                let expected = vec![1, 2, 0, 4, 3];
                assert_eq!(inverse_permutation_helper(5, vec![input])?, expected);
            }
            // malformed input
            {
                let input = Value::from_flattened_array(&[2, 0, 1, 4, 4], UINT64)?;
                let e = inverse_permutation_helper(5, vec![input]);
                assert!(e.is_err());
            }
            {
                let input = Value::from_flattened_array(&[2, 0, 1, 4, 5], UINT64)?;
                let e = inverse_permutation_helper(5, vec![input]);
                assert!(e.is_err());
            }
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_gather() {
        || -> Result<()> {
            {
                // [5]-array
                let input = Value::from_flattened_array(&[1, 2, 3, 4, 5], UINT32)?;
                // [3]-array
                let indices = Value::from_flattened_array(&[2, 0, 4], UINT64)?;
                // output [3]-array
                let expected = vec![3, 1, 5];
                assert_eq!(
                    gather_helper(vec![5], vec![3], 0, vec![input, indices])?,
                    expected
                );
            }
            {
                // [3]-array
                let input = Value::from_flattened_array(&[1, 2, 3], UINT32)?;
                // [3]-array
                let indices = Value::from_flattened_array(&[2, 0, 1], UINT64)?;
                // output [3]-array
                let expected = vec![3, 1, 2];
                assert_eq!(
                    gather_helper(vec![3], vec![3], 0, vec![input, indices])?,
                    expected
                );
            }
            {
                // [2,3,2]-array
                let input =
                    Value::from_flattened_array(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], UINT32)?;
                // [2]-array
                let indices = Value::from_flattened_array(&[2, 0], UINT64)?;
                // output [2,2,2]-array
                let expected = vec![5, 6, 1, 2, 11, 12, 7, 8];
                assert_eq!(
                    gather_helper(vec![2, 3, 2], vec![2], 1, vec![input, indices])?,
                    expected
                );
            }
            {
                let mut input_entries = vec![];
                for i in 1..=20 {
                    input_entries.push(i);
                }
                // [2,5,2]-array
                let input = Value::from_flattened_array(&input_entries, UINT32)?;
                // [2,2]-array
                let indices = Value::from_flattened_array(&[1, 0, 2, 4], UINT64)?;
                // output [2,2,2,2]-array
                let expected = vec![3, 4, 1, 2, 5, 6, 9, 10, 13, 14, 11, 12, 15, 16, 19, 20];
                assert_eq!(
                    gather_helper(vec![2, 5, 2], vec![2, 2], 1, vec![input, indices])?,
                    expected
                );
            }
            {
                // [5]-array
                let input = Value::from_flattened_array(&[1, 2, 3, 4, 5], UINT32)?;
                // [3]-array
                let indices = Value::from_flattened_array(&[2, 0, 0], UINT64)?;
                // [3]-array
                let expected = vec![3, 1, 1];
                assert_eq!(
                    gather_helper(vec![5], vec![3], 0, vec![input, indices])?,
                    expected
                );
            }
            {
                // [5]-array
                let input = Value::from_flattened_array(&[1, 2, 3, 4, 5], UINT32)?;
                // [3]-array
                let indices = Value::from_flattened_array(&[2, 5, 0], UINT64)?;
                let e = gather_helper(vec![5], vec![3], 0, vec![input, indices]);
                assert!(e.is_err());
            }
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_apply_permutation() -> Result<()> {
        let helper = |inputs: Vec<TypedValue>| -> Result<Vec<u64>> {
            let c = simple_context(|g| {
                let inp = g.input(inputs[0].t.clone())?;
                let permutation = g.input(inputs[1].t.clone())?;
                inp.apply_permutation(permutation)
            })?;
            let g = c.get_main_graph()?;
            let o = g.get_output_node()?;
            let result_value = random_evaluate(g, inputs.into_iter().map(|tv| tv.value).collect())?;
            let result_type = o.get_type()?;
            result_value.to_flattened_array_u64(result_type)
        };
        let input = TypedValue::new(
            array_type(vec![5], UINT32),
            Value::from_flattened_array(&[1, 2, 3, 4, 5], UINT32)?,
        )?;
        // UINT16
        let permutation = TypedValue::new(
            array_type(vec![5], UINT16),
            Value::from_flattened_array(&[0, 4, 2, 1, 3], UINT16)?,
        )?;
        let expected = vec![1, 5, 3, 2, 4];
        assert_eq!(helper(vec![input.clone(), permutation])?, expected);
        // UINT64
        let permutation = TypedValue::new(
            array_type(vec![5], UINT64),
            Value::from_flattened_array(&[0, 4, 2, 1, 3], UINT64)?,
        )?;
        let expected = vec![1, 5, 3, 2, 4];
        assert_eq!(helper(vec![input.clone(), permutation])?, expected);

        // Not a valid permutation.
        let permutation = TypedValue::new(
            array_type(vec![5], UINT64),
            Value::from_flattened_array(&[5, 4, 2, 1, 3], UINT64)?,
        )?;
        assert!(helper(vec![input.clone(), permutation]).is_err());

        // Permutation type must be unsigned.
        let permutation = TypedValue::new(
            array_type(vec![5], INT64),
            Value::from_flattened_array(&[5, 4, 2, 1, 3], INT64)?,
        )?;
        assert!(helper(vec![input, permutation]).is_err());

        // [3,2,2]-array
        let input = TypedValue::new(
            array_type(vec![3, 2, 2], UINT32),
            Value::from_flattened_array(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], UINT32)?,
        )?;
        // [3]-array
        let permutation = TypedValue::new(
            array_type(vec![3], UINT32),
            Value::from_flattened_array(&[2, 0, 1], UINT32)?,
        )?;
        // output [3,2,2]-array
        let expected = vec![9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(helper(vec![input, permutation])?, expected);
        Ok(())
    }

    #[test]
    fn test_apply_inverse_permutation() -> Result<()> {
        let helper = |input_shape: Vec<u64>,
                      permutation_shape: Vec<u64>,
                      inputs: Vec<Value>|
         -> Result<Vec<u64>> {
            let c = simple_context(|g| {
                let inp = g.input(array_type(input_shape.clone(), UINT32))?;
                let permutation = g.input(array_type(permutation_shape.clone(), UINT64))?;
                inp.apply_inverse_permutation(permutation)
            })?;
            let g = c.get_main_graph()?;
            let o = g.get_output_node()?;
            let result_value = random_evaluate(g, inputs)?;
            let result_type = o.get_type()?;
            result_value.to_flattened_array_u64(result_type)
        };
        let input = Value::from_flattened_array(&[1, 2, 3, 4, 5], UINT32)?;
        let permutation = Value::from_flattened_array(&[0, 4, 2, 1, 3], UINT64)?;
        let expected = vec![1, 4, 3, 5, 2];
        assert_eq!(
            helper(vec![5], vec![5], vec![input, permutation])?,
            expected
        );

        // [4,3]-array
        let input = Value::from_flattened_array(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], UINT32)?;
        // [4]-array
        let permutation = Value::from_flattened_array(&[2, 0, 3, 1], UINT64)?;
        // output [4,3]-array
        let expected = vec![4, 5, 6, 10, 11, 12, 1, 2, 3, 7, 8, 9];
        assert_eq!(
            helper(vec![4, 3], vec![4], vec![input, permutation])?,
            expected
        );
        Ok(())
    }

    fn random_permutation_helper(n: u64) -> Result<()> {
        let c = simple_context(|g| g.random_permutation(n))?;
        let g = c.get_main_graph()?;
        let o = g.get_output_node()?;
        let result_type = o.get_type()?;

        let mut perm_statistics: HashMap<Vec<u64>, u64> = HashMap::new();
        let expected_count_per_perm = 100;
        let n_factorial: u64 = (2..=n).product();
        let runs = expected_count_per_perm * n_factorial;
        for _ in 0..runs {
            let result_value = random_evaluate(g.clone(), vec![])?;
            let perm = result_value.to_flattened_array_u64(result_type.clone())?;

            let mut perm_sorted = perm.clone();
            perm_sorted.sort();
            let range_vec: Vec<u64> = (0..n).collect();
            assert_eq!(perm_sorted, range_vec);

            perm_statistics
                .entry(perm)
                .and_modify(|counter| *counter += 1)
                .or_insert(0);
        }

        // Check that all permutations occurred in the experiments
        assert_eq!(perm_statistics.len() as u64, n_factorial);

        // Chi-square test with significance level 10^(-6)
        // <https://www.itl.nist.gov/div898/handbook/eda/section3/eda35f.htm>
        if n > 1 {
            let counters: Vec<u64> = perm_statistics.values().map(|c| *c).collect();
            let chi2 = chi_statistics(&counters, expected_count_per_perm);
            // Critical value is computed with n!-1 degrees of freedom
            if n == 4 {
                assert!(chi2 < 70.5496_f64);
            }
            if n == 5 {
                assert!(chi2 < 207.1986_f64);
            }
        }

        Ok(())
    }

    #[test]
    fn test_random_permutation() {
        || -> Result<()> {
            random_permutation_helper(1)?;
            random_permutation_helper(4)?;
            random_permutation_helper(5)?;

            Ok(())
        }()
        .unwrap();
    }

    fn cuckoo_to_permutation_helper(
        shape: ArrayShape,
        input_value: Value,
        seed: Option<[u8; 16]>,
    ) -> Result<Vec<u64>> {
        let input_type = array_type(shape, UINT64);
        let c = simple_context(|g| {
            let i = g.input(input_type.clone())?;
            i.cuckoo_to_permutation()
        })?;
        let g = c.get_main_graph()?;
        let result_value = evaluate_simple_evaluator(g, vec![input_value], seed)?;
        result_value.to_flattened_array_u64(input_type)
    }

    #[test]
    fn test_cuckoo_to_permutation() {
        || -> Result<()> {
            let seed = Some([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
            let x = CUCKOO_DUMMY_ELEMENT;
            // fixed seed
            {
                let input_value = Value::from_flattened_array(&[0, 1, 2, 3], UINT64)?;
                let expected = vec![0, 1, 2, 3];
                assert_eq!(
                    cuckoo_to_permutation_helper(vec![4], input_value, seed)?,
                    expected
                );
            }
            {
                let input_value = Value::from_flattened_array(&[0, x, 2, 1], UINT64)?;
                let expected = vec![0, 3, 2, 1];
                assert_eq!(
                    cuckoo_to_permutation_helper(vec![4], input_value, seed)?,
                    expected
                );
            }
            {
                let input_value = Value::from_flattened_array(&[0, x, 2, 1, x, 3, 4, x], UINT64)?;
                let expected = vec![0, 6, 2, 1, 5, 3, 4, 7];
                assert_eq!(
                    cuckoo_to_permutation_helper(vec![8], input_value, seed)?,
                    expected
                );
            }
            {
                let input_value = Value::from_flattened_array(&[0, x, 2, 1, x, 0, 1, x], UINT64)?;
                let expected = vec![0, 3, 2, 1, 2, 0, 1, 3];
                assert_eq!(
                    cuckoo_to_permutation_helper(vec![2, 4], input_value, seed)?,
                    expected
                );
            }
            {
                let input_value = Value::from_flattened_array(&[0, x, 2, 1, x, 4, 4, x], UINT64)?;
                let e = cuckoo_to_permutation_helper(vec![8], input_value, seed);
                assert!(e.is_err());
            }
            {
                let input_value = Value::from_flattened_array(&[0, x, 2, 1, x, 5, 4, x], UINT64)?;
                let e = cuckoo_to_permutation_helper(vec![8], input_value, seed);
                assert!(e.is_err());
            }
            // random seed
            {
                let input_array = vec![0, x, 2, 1, x, x, 3, x];
                let max_element = 3;
                let input_value = Value::from_flattened_array(&input_array, UINT64)?;
                let mut perm_statistics: HashMap<Vec<u64>, u64> = HashMap::new();
                let expected_count_per_perm = 100;
                let n = 4;
                let n_factorial = (2..=n).product::<u64>();
                let runs = expected_count_per_perm * n_factorial;
                for _ in 0..runs {
                    let res = cuckoo_to_permutation_helper(vec![8], input_value.clone(), None)?;

                    // Extract generated random indices
                    let mut perm = vec![];
                    for i in res {
                        if i > max_element {
                            perm.push(i);
                        }
                    }

                    let mut perm_without_dup = perm.clone();
                    perm_without_dup.sort_unstable();
                    perm_without_dup.dedup();
                    assert_eq!(perm.len(), perm_without_dup.len());

                    let mut perm_sorted = perm.clone();
                    perm_sorted.sort();
                    let range_vec: Vec<u64> = (max_element + 1..input_array.len() as u64).collect();
                    assert_eq!(perm_sorted, range_vec);

                    perm_statistics
                        .entry(perm)
                        .and_modify(|counter| *counter += 1)
                        .or_insert(0);
                }

                // Check that all permutations occurred in the experiments
                assert_eq!(perm_statistics.len() as u64, n_factorial);

                // Chi-square test with significance level 10^(-6)
                // Critical value is computed with n!-1 degrees of freedom
                // <https://www.itl.nist.gov/div898/handbook/eda/section3/eda35f.htm>
                let counters: Vec<u64> = perm_statistics.values().map(|c| *c).collect();
                let chi2 = chi_statistics(&counters, expected_count_per_perm);

                assert!(chi2 < 70.5496_f64);
            }
            Ok(())
        }()
        .unwrap();
    }

    fn decompose_switching_map_helper(
        shape: ArrayShape,
        n: u64,
        input_value: Value,
        seed: Option<[u8; 16]>,
    ) -> Result<(Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>)> {
        let input_type = array_type(shape.clone(), UINT64);
        let c = simple_context(|g| {
            let i = g.input(input_type.clone())?;
            i.decompose_switching_map(n)
        })?;
        let g = c.get_main_graph()?;
        let result_vector = evaluate_simple_evaluator(g, vec![input_value], seed)?.to_vector()?;

        let perm1 = result_vector[0].to_flattened_array_u64(input_type.clone())?;
        let dup_tuple = result_vector[1].to_vector()?;
        let dup_map = dup_tuple[0].to_flattened_array_u64(array_type(shape.clone(), UINT64))?;
        let dup_bits = dup_tuple[1].to_flattened_array_u64(array_type(shape, BIT))?;
        let perm2 = result_vector[2].to_flattened_array_u64(input_type.clone())?;

        Ok((perm1, dup_map, dup_bits, perm2))
    }

    fn compose_maps(
        perm1: &[u64],
        duplication_map: &[u64],
        duplication_bits: &[u64],
        perm2: &[u64],
    ) -> Result<Vec<u64>> {
        let mut result = vec![0; perm1.len()];

        let mut duplication_indices_map = vec![0; duplication_map.len()];

        for i in 1..duplication_bits.len() {
            let bit = duplication_bits[i];
            duplication_indices_map[i] =
                bit * duplication_indices_map[i - 1] + (1 - bit) * i as u64;
        }
        assert_eq!(duplication_map, &duplication_indices_map);

        for i in 0..perm2.len() {
            result[i] = perm1[duplication_map[perm2[i] as usize] as usize];
        }

        Ok(result)
    }

    #[test]
    fn test_decompose_switching_map() {
        || -> Result<()> {
            let seed = Some([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);

            let helper = |switching_map: &[u64],
                          n: u64,
                          expected_perm1: &[u64],
                          expected_dup_bits: &[u64],
                          expected_perm2: &[u64]|
             -> Result<()> {
                let mut expected_dup_map = vec![];
                for i in 0..expected_perm1.len() {
                    if expected_dup_bits[i] == 1 {
                        expected_dup_map.push(expected_dup_map[i - 1]);
                    } else {
                        expected_dup_map.push(i as u64);
                    }
                }

                let input_value = Value::from_flattened_array(switching_map, UINT64)?;
                let (res_perm1, res_dup_map, res_dup_bits, res_perm2) =
                    decompose_switching_map_helper(
                        vec![switching_map.len() as u64],
                        n,
                        input_value,
                        seed,
                    )?;

                assert_eq!(
                    (
                        &res_perm1[..],
                        &res_dup_map[..],
                        &res_dup_bits[..],
                        &res_perm2[..]
                    ),
                    (
                        expected_perm1,
                        &expected_dup_map[..],
                        expected_dup_bits,
                        expected_perm2
                    )
                );

                let res_composition =
                    compose_maps(&res_perm1, &res_dup_map, &res_dup_bits, &res_perm2)?;
                assert_eq!(&res_composition, switching_map);

                Ok(())
            };

            // fixed seed
            {
                let input_map = vec![2, 0, 1, 3, 2, 4, 3, 8];

                let expected_perm1 = vec![2, 6, 0, 1, 3, 5, 4, 8];
                let expected_dup_map = vec![0, 1, 0, 0, 0, 1, 0, 0];
                let expected_perm2 = vec![0, 2, 3, 4, 1, 6, 5, 7];

                helper(
                    &input_map,
                    9,
                    &expected_perm1,
                    &expected_dup_map,
                    &expected_perm2,
                )?;
            }
            {
                let input_map = vec![0, 1, 2, 3, 4, 5, 6];
                let expected_perm1 = vec![0, 1, 2, 3, 4, 5, 6];
                let expected_dup_map = vec![0; 7];
                let expected_perm2 = vec![0, 1, 2, 3, 4, 5, 6];

                helper(
                    &input_map,
                    7,
                    &expected_perm1,
                    &expected_dup_map,
                    &expected_perm2,
                )?;
            }
            {
                let input_map = vec![6, 6, 6, 6, 6, 6, 6];
                let expected_perm1 = vec![6, 0, 1, 3, 4, 2, 5];
                let expected_dup_map = vec![0, 1, 1, 1, 1, 1, 1];
                let expected_perm2 = vec![0, 1, 2, 3, 4, 5, 6];

                helper(
                    &input_map,
                    7,
                    &expected_perm1,
                    &expected_dup_map,
                    &expected_perm2,
                )?;
            }
            {
                let input_map = Value::from_flattened_array(&[0, 1, 5], UINT64)?;
                let e = decompose_switching_map_helper(vec![3], 5, input_map, seed);
                assert!(e.is_err());
            }
            // random seed
            {
                let input_array = vec![0, 2, 2, 1, 3, 1, 3, 2];
                let input_value = Value::from_flattened_array(&input_array, UINT64)?;
                let mut perm_statistics: HashMap<Vec<u64>, u64> = HashMap::new();
                let expected_count_per_perm = 100;
                let random_indices = 4;
                let random_indices_factorial = (2..=random_indices).product::<u64>();
                let runs = expected_count_per_perm * random_indices_factorial;
                let n = input_array.len() as u64;
                for _ in 0..runs {
                    let (res_perm1, res_dup_map, res_dup_bits, res_perm2) =
                        decompose_switching_map_helper(vec![n], n, input_value.clone(), None)?;

                    let res_composition =
                        compose_maps(&res_perm1, &res_dup_map, &res_dup_bits, &res_perm2)?;
                    assert_eq!(res_composition, input_array);

                    let mut perm_sorted = res_perm1.clone();
                    perm_sorted.sort();
                    let range_vec: Vec<u64> = (0..n).collect();
                    assert_eq!(perm_sorted, range_vec);

                    perm_statistics
                        .entry(res_perm1)
                        .and_modify(|counter| *counter += 1)
                        .or_insert(0);
                }

                // Check that all permutations occurred in the experiments
                assert_eq!(perm_statistics.len() as u64, random_indices_factorial);

                // Chi-square test with significance level 10^(-6)
                // Critical value is computed with n!-1 degrees of freedom
                // <https://www.itl.nist.gov/div898/handbook/eda/section3/eda35f.htm>
                let counters: Vec<u64> = perm_statistics.values().map(|c| *c).collect();
                let chi2 = chi_statistics(&counters, expected_count_per_perm);

                assert!(chi2 < 70.5496_f64);
            }

            Ok(())
        }()
        .unwrap();
    }

    fn join_helper(test_info: Vec<JoinTestInfo>, join_t: JoinType) -> Result<()> {
        for test_i in test_info {
            let c = simple_context(|g| {
                let i0 = g.input(test_i.set0.get_type())?;
                let i1 = g.input(test_i.set1.get_type())?;
                i0.join(i1, join_t, test_i.headers)
            })?;
            let g = c.get_main_graph()?;
            let o = g.get_output_node()?;
            let result =
                random_evaluate(g, vec![test_i.set0.get_value()?, test_i.set1.get_value()?])?
                    .to_vector()?;
            let result_t = o.get_type()?;
            if let Type::NamedTuple(headers_types) = result_t {
                assert_eq!(test_i.expected.len(), headers_types.len());
                for (i, (expected_header, expected_column)) in test_i.expected.iter().enumerate() {
                    assert_eq!(*expected_header, headers_types[i].0);
                    assert_eq!(
                        result[i].to_flattened_array_u64((*headers_types[i].1).clone())?,
                        *expected_column
                    )
                }
            } else {
                panic!("Inconsistency with type checker");
            }
        }
        Ok(())
    }

    struct ColumnInfo {
        header: String,
        shape: Vec<u64>,
        st: ScalarType,
        data: Vec<u64>,
    }

    fn column_info(header: &str, shape: &[u64], st: ScalarType, data: &[u64]) -> ColumnInfo {
        ColumnInfo {
            header: header.to_owned(),
            shape: shape.to_vec(),
            st,
            data: data.to_vec(),
        }
    }

    type SetInfo = Vec<ColumnInfo>;

    trait SetHelpers {
        fn get_type(&self) -> Type;
        fn get_value(&self) -> Result<Value>;
    }

    impl SetHelpers for SetInfo {
        fn get_type(&self) -> Type {
            let mut v = vec![];
            for col in self.iter() {
                v.push((
                    col.header.clone(),
                    array_type(col.shape.clone(), col.st.clone()),
                ));
            }
            named_tuple_type(v)
        }
        fn get_value(&self) -> Result<Value> {
            let mut v = vec![];
            for col in self.iter() {
                v.push(Value::from_flattened_array(&col.data, col.st.clone())?);
            }
            Ok(Value::from_vector(v))
        }
    }

    type ExpectedInfo = Vec<(String, Vec<u64>)>;

    fn expected_info(expected_columns: Vec<(&str, &[u64])>) -> ExpectedInfo {
        let mut v = vec![];
        for (header, data) in expected_columns {
            v.push((header.to_owned(), data.to_vec()));
        }
        v
    }

    struct JoinTestInfo {
        set0: SetInfo,
        set1: SetInfo,
        headers: HashMap<String, String>,
        expected: ExpectedInfo,
    }

    fn join_info(
        set0: SetInfo,
        set1: SetInfo,
        headers: Vec<(&str, &str)>,
        expected: ExpectedInfo,
    ) -> JoinTestInfo {
        let mut hmap = HashMap::new();
        for (h0, h1) in headers {
            hmap.insert(h0.to_owned(), h1.to_owned());
        }
        JoinTestInfo {
            set0,
            set1,
            headers: hmap,
            expected,
        }
    }

    #[test]
    fn test_set_intersection() -> Result<()> {
        let tests = vec![
            join_info(
                vec![
                    column_info(NULL_HEADER, &[6], BIT, &[1, 1, 1, 1, 1, 1]),
                    column_info("ID", &[6], UINT64, &[5, 3, 0, 4, 1, 2]),
                    column_info("Income", &[6], UINT64, &[500, 300, 0, 400, 100, 200]),
                ],
                vec![
                    column_info(NULL_HEADER, &[10], BIT, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                    column_info("ID", &[10], UINT64, &[4, 7, 8, 9, 10, 11, 12, 2, 3, 13]),
                    column_info(
                        "Outcome",
                        &[10],
                        UINT64,
                        &[40, 70, 80, 90, 100, 110, 120, 20, 30, 130],
                    ),
                ],
                vec![("ID", "ID")],
                expected_info(vec![
                    (NULL_HEADER, &[0, 1, 0, 1, 0, 1]),
                    ("ID", &[0, 3, 0, 4, 0, 2]),
                    ("Income", &[0, 300, 0, 400, 0, 200]),
                    ("Outcome", &[0, 30, 0, 40, 0, 20]),
                ]),
            ),
            join_info(
                vec![
                    column_info(NULL_HEADER, &[6], BIT, &[1, 1, 1, 1, 1, 1]),
                    column_info("ID", &[6], UINT64, &[5, 3, 0, 4, 1, 2]),
                    column_info("Income1", &[6], UINT64, &[50, 30, 0, 40, 10, 20]),
                ],
                vec![
                    column_info(NULL_HEADER, &[10], BIT, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                    column_info("ID", &[10], UINT64, &[4, 7, 8, 9, 10, 11, 12, 2, 3, 13]),
                    column_info(
                        "Income2",
                        &[10],
                        UINT64,
                        &[40, 70, 80, 90, 100, 110, 120, 20, 30, 130],
                    ),
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                expected_info(vec![
                    (NULL_HEADER, &[0, 1, 0, 1, 0, 1]),
                    ("ID", &[0, 3, 0, 4, 0, 2]),
                    ("Income1", &[0, 30, 0, 40, 0, 20]),
                ]),
            ),
            join_info(
                vec![
                    column_info(NULL_HEADER, &[6], BIT, &[1, 1, 1, 1, 1, 0]),
                    column_info("ID", &[6], UINT64, &[5, 3, 0, 4, 1, 2]),
                    column_info("Income1", &[6], UINT64, &[50, 30, 0, 40, 10, 20]),
                    column_info("Outcome1", &[6], UINT64, &[500, 300, 0, 400, 100, 200]),
                ],
                vec![
                    column_info(NULL_HEADER, &[10], BIT, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                    column_info("ID", &[10], UINT64, &[4, 7, 8, 9, 10, 11, 12, 2, 3, 13]),
                    column_info(
                        "Income2",
                        &[10],
                        UINT64,
                        &[40, 70, 80, 90, 100, 110, 120, 20, 30, 130],
                    ),
                    column_info(
                        "Outcome2",
                        &[10],
                        UINT64,
                        &[400, 700, 800, 900, 1000, 1100, 1200, 200, 300, 1300],
                    ),
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                expected_info(vec![
                    (NULL_HEADER, &[0, 1, 0, 1, 0, 0]),
                    ("ID", &[0, 3, 0, 4, 0, 0]),
                    ("Income1", &[0, 30, 0, 40, 0, 0]),
                    ("Outcome1", &[0, 300, 0, 400, 0, 0]),
                    ("Outcome2", &[0, 300, 0, 400, 0, 0]),
                ]),
            ),
            join_info(
                vec![
                    column_info(NULL_HEADER, &[6], BIT, &[1, 0, 1, 0, 1, 1]),
                    column_info("Income1", &[6], UINT64, &[5, 3, 0, 4, 1, 2]),
                    column_info("ID", &[6], UINT64, &[50, 30, 0, 40, 10, 20]),
                    column_info("Outcome1", &[6], UINT64, &[500, 300, 0, 400, 100, 200]),
                ],
                vec![
                    column_info(NULL_HEADER, &[10], BIT, &[1, 1, 1, 1, 1, 1, 1, 0, 1, 1]),
                    column_info("ID", &[10], UINT64, &[4, 7, 8, 9, 10, 11, 12, 2, 3, 13]),
                    column_info(
                        "Income2",
                        &[10],
                        UINT64,
                        &[40, 70, 80, 90, 100, 110, 120, 20, 30, 130],
                    ),
                    column_info(
                        "Outcome2",
                        &[10],
                        UINT64,
                        &[400, 700, 800, 900, 1000, 1100, 1200, 200, 300, 1300],
                    ),
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                expected_info(vec![
                    (NULL_HEADER, &[0, 0, 0, 0, 0, 0]),
                    ("Income1", &[0, 0, 0, 0, 0, 0]),
                    ("ID", &[0, 0, 0, 0, 0, 0]),
                    ("Outcome1", &[0, 0, 0, 0, 0, 0]),
                    ("Outcome2", &[0, 0, 0, 0, 0, 0]),
                ]),
            ),
            join_info(
                vec![
                    column_info(NULL_HEADER, &[1], BIT, &[1]),
                    column_info("ID", &[1], UINT64, &[5]),
                    column_info("Income1", &[1], UINT64, &[50]),
                    column_info("Outcome1", &[1], UINT64, &[500]),
                ],
                vec![
                    column_info(NULL_HEADER, &[1], BIT, &[1]),
                    column_info("ID", &[1], UINT64, &[5]),
                    column_info("Income2", &[1], UINT64, &[50]),
                    column_info("Outcome2", &[1], UINT64, &[51]),
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                expected_info(vec![
                    (NULL_HEADER, &[1]),
                    ("ID", &[5]),
                    ("Income1", &[50]),
                    ("Outcome1", &[500]),
                    ("Outcome2", &[51]),
                ]),
            ),
            join_info(
                vec![
                    column_info(NULL_HEADER, &[1], BIT, &[1]),
                    column_info("Income1", &[1], UINT64, &[50]),
                    column_info("Outcome1", &[1], UINT64, &[500]),
                    column_info("ID", &[1], UINT64, &[5]),
                ],
                vec![
                    column_info(NULL_HEADER, &[1], BIT, &[1]),
                    column_info("ID", &[1], UINT64, &[5]),
                    column_info("Income2", &[1], UINT64, &[50]),
                    column_info("Outcome2", &[1], UINT64, &[51]),
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                expected_info(vec![
                    (NULL_HEADER, &[1]),
                    ("Income1", &[50]),
                    ("Outcome1", &[500]),
                    ("ID", &[5]),
                    ("Outcome2", &[51]),
                ]),
            ),
            join_info(
                vec![
                    column_info(NULL_HEADER, &[2], BIT, &[1, 1]),
                    column_info("Income1", &[2], UINT64, &[40, 50]),
                    column_info("Outcome1", &[2, 2], UINT64, &[400, 401, 500, 501]),
                    column_info("ID", &[2], UINT64, &[4, 5]),
                ],
                vec![
                    column_info(NULL_HEADER, &[2], BIT, &[1, 1]),
                    column_info("ID", &[2], UINT64, &[5, 3]),
                    column_info("Income2", &[2], UINT64, &[50, 30]),
                    column_info("Outcome2", &[2, 2], UINT64, &[500, 501, 300, 301]),
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                expected_info(vec![
                    (NULL_HEADER, &[0, 1]),
                    ("Income1", &[0, 50]),
                    ("Outcome1", &[0, 0, 500, 501]),
                    ("ID", &[0, 5]),
                    ("Outcome2", &[0, 0, 500, 501]),
                ]),
            ),
        ];
        join_helper(tests, JoinType::Inner)?;

        Ok(())
    }

    #[test]
    fn test_left_join() -> Result<()> {
        let tests = vec![
            join_info(
                vec![
                    column_info(NULL_HEADER, &[6], BIT, &[1, 1, 1, 1, 1, 1]),
                    column_info("ID", &[6], UINT64, &[5, 3, 0, 4, 1, 2]),
                    column_info("Income", &[6], UINT64, &[500, 300, 0, 400, 100, 200]),
                ],
                vec![
                    column_info(NULL_HEADER, &[10], BIT, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                    column_info("ID", &[10], UINT64, &[4, 7, 8, 9, 10, 11, 12, 2, 3, 13]),
                    column_info(
                        "Outcome",
                        &[10],
                        UINT64,
                        &[40, 70, 80, 90, 100, 110, 120, 20, 30, 130],
                    ),
                ],
                vec![("ID", "ID")],
                expected_info(vec![
                    (NULL_HEADER, &[1, 1, 1, 1, 1, 1]),
                    ("ID", &[5, 3, 0, 4, 1, 2]),
                    ("Income", &[500, 300, 0, 400, 100, 200]),
                    ("Outcome", &[0, 30, 0, 40, 0, 20]),
                ]),
            ),
            join_info(
                vec![
                    column_info(NULL_HEADER, &[6], BIT, &[1, 1, 1, 1, 1, 1]),
                    column_info("ID", &[6], UINT64, &[5, 3, 0, 4, 1, 2]),
                    column_info("Income1", &[6], UINT64, &[50, 30, 0, 40, 10, 20]),
                ],
                vec![
                    column_info(NULL_HEADER, &[10], BIT, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                    column_info("ID", &[10], UINT64, &[4, 7, 8, 9, 10, 11, 12, 2, 3, 13]),
                    column_info(
                        "Income2",
                        &[10],
                        UINT64,
                        &[40, 70, 80, 90, 100, 110, 120, 20, 30, 130],
                    ),
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                expected_info(vec![
                    (NULL_HEADER, &[1, 1, 1, 1, 1, 1]),
                    ("ID", &[5, 3, 0, 4, 1, 2]),
                    ("Income1", &[50, 30, 0, 40, 10, 20]),
                ]),
            ),
            join_info(
                vec![
                    column_info(NULL_HEADER, &[6], BIT, &[1, 1, 1, 1, 1, 0]),
                    column_info("ID", &[6], UINT64, &[5, 3, 0, 4, 1, 2]),
                    column_info("Income1", &[6], UINT64, &[50, 30, 0, 40, 10, 20]),
                    column_info("Outcome1", &[6], UINT64, &[500, 300, 0, 400, 100, 200]),
                ],
                vec![
                    column_info(NULL_HEADER, &[10], BIT, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                    column_info("ID", &[10], UINT64, &[4, 7, 8, 9, 10, 11, 12, 2, 3, 13]),
                    column_info(
                        "Income2",
                        &[10],
                        UINT64,
                        &[40, 70, 80, 90, 100, 110, 120, 20, 30, 130],
                    ),
                    column_info(
                        "Outcome2",
                        &[10],
                        UINT64,
                        &[400, 700, 800, 900, 1000, 1100, 1200, 200, 300, 1300],
                    ),
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                expected_info(vec![
                    (NULL_HEADER, &[1, 1, 1, 1, 1, 0]),
                    ("ID", &[5, 3, 0, 4, 1, 0]),
                    ("Income1", &[50, 30, 0, 40, 10, 0]),
                    ("Outcome1", &[500, 300, 0, 400, 100, 0]),
                    ("Outcome2", &[0, 300, 0, 400, 0, 0]),
                ]),
            ),
            join_info(
                vec![
                    column_info(NULL_HEADER, &[6], BIT, &[1, 0, 1, 0, 1, 1]),
                    column_info("Income1", &[6], UINT64, &[5, 3, 0, 4, 1, 2]),
                    column_info("ID", &[6], UINT64, &[50, 30, 0, 40, 10, 20]),
                    column_info("Outcome1", &[6], UINT64, &[500, 300, 0, 400, 100, 200]),
                ],
                vec![
                    column_info(NULL_HEADER, &[10], BIT, &[1, 1, 1, 1, 1, 1, 1, 0, 1, 1]),
                    column_info("ID", &[10], UINT64, &[4, 7, 8, 9, 10, 11, 12, 2, 3, 13]),
                    column_info(
                        "Income2",
                        &[10],
                        UINT64,
                        &[40, 70, 80, 90, 100, 110, 120, 20, 30, 130],
                    ),
                    column_info(
                        "Outcome2",
                        &[10],
                        UINT64,
                        &[400, 700, 800, 900, 1000, 1100, 1200, 200, 300, 1300],
                    ),
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                expected_info(vec![
                    (NULL_HEADER, &[1, 0, 1, 0, 1, 1]),
                    ("Income1", &[5, 0, 0, 0, 1, 2]),
                    ("ID", &[50, 0, 0, 0, 10, 20]),
                    ("Outcome1", &[500, 0, 0, 0, 100, 200]),
                    ("Outcome2", &[0, 0, 0, 0, 0, 0]),
                ]),
            ),
            join_info(
                vec![
                    column_info(NULL_HEADER, &[1], BIT, &[1]),
                    column_info("ID", &[1], UINT64, &[5]),
                    column_info("Income1", &[1], UINT64, &[50]),
                    column_info("Outcome1", &[1], UINT64, &[500]),
                ],
                vec![
                    column_info(NULL_HEADER, &[1], BIT, &[1]),
                    column_info("ID", &[1], UINT64, &[5]),
                    column_info("Income2", &[1], UINT64, &[50]),
                    column_info("Outcome2", &[1], UINT64, &[51]),
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                expected_info(vec![
                    (NULL_HEADER, &[1]),
                    ("ID", &[5]),
                    ("Income1", &[50]),
                    ("Outcome1", &[500]),
                    ("Outcome2", &[51]),
                ]),
            ),
            join_info(
                vec![
                    column_info(NULL_HEADER, &[1], BIT, &[1]),
                    column_info("Income1", &[1], UINT64, &[50]),
                    column_info("Outcome1", &[1], UINT64, &[500]),
                    column_info("ID", &[1], UINT64, &[5]),
                ],
                vec![
                    column_info(NULL_HEADER, &[1], BIT, &[1]),
                    column_info("ID", &[1], UINT64, &[5]),
                    column_info("Income2", &[1], UINT64, &[50]),
                    column_info("Outcome2", &[1], UINT64, &[51]),
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                expected_info(vec![
                    (NULL_HEADER, &[1]),
                    ("Income1", &[50]),
                    ("Outcome1", &[500]),
                    ("ID", &[5]),
                    ("Outcome2", &[51]),
                ]),
            ),
            join_info(
                vec![
                    column_info(NULL_HEADER, &[2], BIT, &[1, 1]),
                    column_info("Income1", &[2], UINT64, &[40, 50]),
                    column_info("Outcome1", &[2, 2], UINT64, &[400, 401, 500, 501]),
                    column_info("ID", &[2], UINT64, &[4, 5]),
                ],
                vec![
                    column_info(NULL_HEADER, &[2], BIT, &[1, 1]),
                    column_info("ID", &[2], UINT64, &[5, 3]),
                    column_info("Income2", &[2], UINT64, &[50, 30]),
                    column_info("Outcome2", &[2, 2], UINT64, &[500, 501, 300, 301]),
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                expected_info(vec![
                    (NULL_HEADER, &[1, 1]),
                    ("Income1", &[40, 50]),
                    ("Outcome1", &[400, 401, 500, 501]),
                    ("ID", &[4, 5]),
                    ("Outcome2", &[0, 0, 500, 501]),
                ]),
            ),
        ];
        join_helper(tests, JoinType::Left)?;

        Ok(())
    }

    #[test]
    fn test_union_join() -> Result<()> {
        let tests = vec![
            join_info(
                vec![
                    column_info(NULL_HEADER, &[6], BIT, &[1, 1, 1, 1, 1, 1]),
                    column_info("ID", &[6], UINT64, &[5, 3, 0, 4, 1, 2]),
                    column_info("Income", &[6], UINT64, &[500, 300, 0, 400, 100, 200]),
                ],
                vec![
                    column_info(NULL_HEADER, &[10], BIT, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                    column_info("ID", &[10], UINT64, &[4, 7, 8, 9, 10, 11, 12, 2, 3, 13]),
                    column_info(
                        "Outcome",
                        &[10],
                        UINT64,
                        &[40, 70, 80, 90, 100, 110, 120, 20, 30, 130],
                    ),
                ],
                vec![("ID", "ID")],
                expected_info(vec![
                    (
                        NULL_HEADER,
                        &[1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    ),
                    ("ID", &[5, 0, 0, 0, 1, 0, 4, 7, 8, 9, 10, 11, 12, 2, 3, 13]),
                    (
                        "Income",
                        &[500, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ),
                    (
                        "Outcome",
                        &[0, 0, 0, 0, 0, 0, 40, 70, 80, 90, 100, 110, 120, 20, 30, 130],
                    ),
                ]),
            ),
            join_info(
                vec![
                    column_info(NULL_HEADER, &[6], BIT, &[1, 1, 1, 1, 1, 1]),
                    column_info("ID", &[6], UINT64, &[5, 3, 0, 4, 1, 2]),
                    column_info("Income1", &[6], UINT64, &[50, 30, 0, 40, 10, 20]),
                ],
                vec![
                    column_info(NULL_HEADER, &[10], BIT, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                    column_info("ID", &[10], UINT64, &[4, 7, 8, 9, 10, 11, 12, 2, 3, 13]),
                    column_info(
                        "Income2",
                        &[10],
                        UINT64,
                        &[40, 70, 80, 90, 100, 110, 120, 20, 30, 130],
                    ),
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                expected_info(vec![
                    (
                        NULL_HEADER,
                        &[1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    ),
                    ("ID", &[5, 0, 0, 0, 1, 0, 4, 7, 8, 9, 10, 11, 12, 2, 3, 13]),
                    (
                        "Income1",
                        &[
                            50, 0, 0, 0, 10, 0, 40, 70, 80, 90, 100, 110, 120, 20, 30, 130,
                        ],
                    ),
                ]),
            ),
            join_info(
                vec![
                    column_info(NULL_HEADER, &[6], BIT, &[1, 1, 1, 1, 1, 0]),
                    column_info("ID", &[6], UINT64, &[5, 3, 0, 4, 1, 2]),
                    column_info("Income1", &[6], UINT64, &[50, 30, 0, 40, 10, 20]),
                    column_info("Outcome1", &[6], UINT64, &[500, 300, 0, 400, 100, 200]),
                ],
                vec![
                    column_info(NULL_HEADER, &[10], BIT, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                    column_info("ID", &[10], UINT64, &[4, 7, 8, 9, 10, 11, 12, 2, 3, 13]),
                    column_info(
                        "Income2",
                        &[10],
                        UINT64,
                        &[40, 70, 80, 90, 100, 110, 120, 20, 30, 130],
                    ),
                    column_info(
                        "Outcome2",
                        &[10],
                        UINT64,
                        &[400, 700, 800, 900, 1000, 1100, 1200, 200, 300, 1300],
                    ),
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                expected_info(vec![
                    (
                        NULL_HEADER,
                        &[1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    ),
                    ("ID", &[5, 0, 0, 0, 1, 0, 4, 7, 8, 9, 10, 11, 12, 2, 3, 13]),
                    (
                        "Income1",
                        &[
                            50, 0, 0, 0, 10, 0, 40, 70, 80, 90, 100, 110, 120, 20, 30, 130,
                        ],
                    ),
                    (
                        "Outcome1",
                        &[500, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ),
                    (
                        "Outcome2",
                        &[
                            0, 0, 0, 0, 0, 0, 400, 700, 800, 900, 1000, 1100, 1200, 200, 300, 1300,
                        ],
                    ),
                ]),
            ),
            join_info(
                vec![
                    column_info(NULL_HEADER, &[6], BIT, &[1, 0, 1, 0, 1, 1]),
                    column_info("Income1", &[6], UINT64, &[5, 3, 0, 4, 1, 2]),
                    column_info("ID", &[6], UINT64, &[50, 30, 0, 40, 10, 20]),
                    column_info("Outcome1", &[6], UINT64, &[500, 300, 0, 400, 100, 200]),
                ],
                vec![
                    column_info(NULL_HEADER, &[10], BIT, &[1, 1, 1, 1, 1, 1, 1, 0, 1, 1]),
                    column_info("ID", &[10], UINT64, &[4, 7, 8, 9, 10, 11, 12, 2, 3, 13]),
                    column_info(
                        "Income2",
                        &[10],
                        UINT64,
                        &[40, 70, 80, 90, 100, 110, 120, 20, 30, 130],
                    ),
                    column_info(
                        "Outcome2",
                        &[10],
                        UINT64,
                        &[400, 700, 800, 900, 1000, 1100, 1200, 200, 300, 1300],
                    ),
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                expected_info(vec![
                    (
                        NULL_HEADER,
                        &[1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                    ),
                    (
                        "Income1",
                        &[5, 0, 0, 0, 1, 2, 40, 70, 80, 90, 100, 110, 120, 0, 30, 130],
                    ),
                    (
                        "ID",
                        &[50, 0, 0, 0, 10, 20, 4, 7, 8, 9, 10, 11, 12, 0, 3, 13],
                    ),
                    (
                        "Outcome1",
                        &[500, 0, 0, 0, 100, 200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ),
                    (
                        "Outcome2",
                        &[
                            0, 0, 0, 0, 0, 0, 400, 700, 800, 900, 1000, 1100, 1200, 0, 300, 1300,
                        ],
                    ),
                ]),
            ),
            join_info(
                vec![
                    column_info(NULL_HEADER, &[1], BIT, &[1]),
                    column_info("ID", &[1], UINT64, &[5]),
                    column_info("Income1", &[1], UINT64, &[50]),
                    column_info("Outcome1", &[1], UINT64, &[500]),
                ],
                vec![
                    column_info(NULL_HEADER, &[1], BIT, &[1]),
                    column_info("ID", &[1], UINT64, &[5]),
                    column_info("Income2", &[1], UINT64, &[50]),
                    column_info("Outcome2", &[1], UINT64, &[51]),
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                expected_info(vec![
                    (NULL_HEADER, &[0, 1]),
                    ("ID", &[0, 5]),
                    ("Income1", &[0, 50]),
                    ("Outcome1", &[0, 0]),
                    ("Outcome2", &[0, 51]),
                ]),
            ),
            join_info(
                vec![
                    column_info(NULL_HEADER, &[1], BIT, &[1]),
                    column_info("Income1", &[1], UINT64, &[50]),
                    column_info("Outcome1", &[1], UINT64, &[500]),
                    column_info("ID", &[1], UINT64, &[5]),
                ],
                vec![
                    column_info(NULL_HEADER, &[1], BIT, &[1]),
                    column_info("ID", &[1], UINT64, &[5]),
                    column_info("Income2", &[1], UINT64, &[50]),
                    column_info("Outcome2", &[1], UINT64, &[51]),
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                expected_info(vec![
                    (NULL_HEADER, &[0, 1]),
                    ("Income1", &[0, 50]),
                    ("Outcome1", &[0, 0]),
                    ("ID", &[0, 5]),
                    ("Outcome2", &[0, 51]),
                ]),
            ),
            join_info(
                vec![
                    column_info(NULL_HEADER, &[2], BIT, &[1, 1]),
                    column_info("Income1", &[2], UINT64, &[40, 50]),
                    column_info("Outcome1", &[2], UINT64, &[400, 500]),
                    column_info("ID", &[2], UINT64, &[4, 5]),
                ],
                vec![
                    column_info(NULL_HEADER, &[2], BIT, &[1, 1]),
                    column_info("ID", &[2], UINT64, &[4, 3]),
                    column_info("Income2", &[2], UINT64, &[40, 30]),
                    column_info("Outcome2", &[2, 2], UINT64, &[40, 41, 30, 31]),
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                expected_info(vec![
                    (NULL_HEADER, &[0, 1, 1, 1]),
                    ("Income1", &[0, 50, 40, 30]),
                    ("Outcome1", &[0, 500, 0, 0]),
                    ("ID", &[0, 5, 4, 3]),
                    ("Outcome2", &[0, 0, 0, 0, 40, 41, 30, 31]),
                ]),
            ),
            join_info(
                vec![
                    column_info(NULL_HEADER, &[2], BIT, &[1, 1]),
                    column_info("Income1", &[2], UINT64, &[40, 50]),
                    column_info("Outcome1", &[2], UINT64, &[400, 500]),
                    column_info("ID", &[2], UINT64, &[4, 5]),
                ],
                vec![
                    column_info(NULL_HEADER, &[2], BIT, &[1, 1]),
                    column_info("ID", &[2], UINT64, &[6, 7]),
                    column_info("Income2", &[2], UINT64, &[60, 70]),
                    column_info("Outcome2", &[2, 2], UINT64, &[60, 61, 70, 71]),
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                expected_info(vec![
                    (NULL_HEADER, &[1, 1, 1, 1]),
                    ("Income1", &[40, 50, 60, 70]),
                    ("Outcome1", &[400, 500, 0, 0]),
                    ("ID", &[4, 5, 6, 7]),
                    ("Outcome2", &[0, 0, 0, 0, 60, 61, 70, 71]),
                ]),
            ),
        ];
        join_helper(tests, JoinType::Union)?;

        Ok(())
    }

    #[test]
    fn test_full_join() -> Result<()> {
        let tests = vec![
            join_info(
                vec![
                    column_info(NULL_HEADER, &[6], BIT, &[1, 1, 1, 1, 1, 1]),
                    column_info("ID", &[6], UINT64, &[5, 3, 0, 4, 1, 2]),
                    column_info("Income", &[6], UINT64, &[500, 300, 0, 400, 100, 200]),
                ],
                vec![
                    column_info(NULL_HEADER, &[10], BIT, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                    column_info("ID", &[10], UINT64, &[4, 7, 8, 9, 10, 11, 12, 2, 3, 13]),
                    column_info(
                        "Outcome",
                        &[10],
                        UINT64,
                        &[40, 70, 80, 90, 100, 110, 120, 20, 30, 130],
                    ),
                ],
                vec![("ID", "ID")],
                expected_info(vec![
                    (
                        NULL_HEADER,
                        &[1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    ),
                    ("ID", &[5, 0, 0, 0, 1, 0, 4, 7, 8, 9, 10, 11, 12, 2, 3, 13]),
                    (
                        "Income",
                        &[500, 0, 0, 0, 100, 0, 400, 0, 0, 0, 0, 0, 0, 200, 300, 0],
                    ),
                    (
                        "Outcome",
                        &[0, 0, 0, 0, 0, 0, 40, 70, 80, 90, 100, 110, 120, 20, 30, 130],
                    ),
                ]),
            ),
            join_info(
                vec![
                    column_info(NULL_HEADER, &[6], BIT, &[1, 1, 1, 1, 1, 1]),
                    column_info("ID", &[6], UINT64, &[5, 3, 0, 4, 1, 2]),
                    column_info("Income1", &[6], UINT64, &[50, 30, 0, 40, 10, 20]),
                ],
                vec![
                    column_info(NULL_HEADER, &[10], BIT, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                    column_info("ID", &[10], UINT64, &[4, 7, 8, 9, 10, 11, 12, 2, 3, 13]),
                    column_info(
                        "Income2",
                        &[10],
                        UINT64,
                        &[40, 70, 80, 90, 100, 110, 120, 20, 30, 130],
                    ),
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                expected_info(vec![
                    (
                        NULL_HEADER,
                        &[1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    ),
                    ("ID", &[5, 0, 0, 0, 1, 0, 4, 7, 8, 9, 10, 11, 12, 2, 3, 13]),
                    (
                        "Income1",
                        &[
                            50, 0, 0, 0, 10, 0, 40, 70, 80, 90, 100, 110, 120, 20, 30, 130,
                        ],
                    ),
                ]),
            ),
            join_info(
                vec![
                    column_info(NULL_HEADER, &[6], BIT, &[1, 1, 1, 1, 1, 0]),
                    column_info("ID", &[6], UINT64, &[5, 3, 0, 4, 1, 2]),
                    column_info("Income1", &[6], UINT64, &[50, 30, 0, 40, 10, 20]),
                    column_info("Outcome1", &[6], UINT64, &[500, 300, 0, 400, 100, 200]),
                ],
                vec![
                    column_info(NULL_HEADER, &[10], BIT, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                    column_info("ID", &[10], UINT64, &[4, 7, 8, 9, 10, 11, 12, 2, 3, 13]),
                    column_info(
                        "Income2",
                        &[10],
                        UINT64,
                        &[40, 70, 80, 90, 100, 110, 120, 20, 30, 130],
                    ),
                    column_info(
                        "Outcome2",
                        &[10],
                        UINT64,
                        &[400, 700, 800, 900, 1000, 1100, 1200, 200, 300, 1300],
                    ),
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                expected_info(vec![
                    (
                        NULL_HEADER,
                        &[1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    ),
                    ("ID", &[5, 0, 0, 0, 1, 0, 4, 7, 8, 9, 10, 11, 12, 2, 3, 13]),
                    (
                        "Income1",
                        &[
                            50, 0, 0, 0, 10, 0, 40, 70, 80, 90, 100, 110, 120, 20, 30, 130,
                        ],
                    ),
                    (
                        "Outcome1",
                        &[500, 0, 0, 0, 100, 0, 400, 0, 0, 0, 0, 0, 0, 0, 300, 0],
                    ),
                    (
                        "Outcome2",
                        &[
                            0, 0, 0, 0, 0, 0, 400, 700, 800, 900, 1000, 1100, 1200, 200, 300, 1300,
                        ],
                    ),
                ]),
            ),
            join_info(
                vec![
                    column_info(NULL_HEADER, &[6], BIT, &[1, 0, 1, 0, 1, 1]),
                    column_info("Income1", &[6], UINT64, &[5, 3, 0, 4, 1, 2]),
                    column_info("ID", &[6], UINT64, &[50, 30, 0, 40, 10, 20]),
                    column_info("Outcome1", &[6], UINT64, &[500, 300, 0, 400, 100, 200]),
                ],
                vec![
                    column_info(NULL_HEADER, &[10], BIT, &[1, 1, 1, 1, 1, 1, 1, 0, 1, 1]),
                    column_info("ID", &[10], UINT64, &[4, 7, 8, 9, 10, 11, 12, 2, 3, 13]),
                    column_info(
                        "Income2",
                        &[10],
                        UINT64,
                        &[40, 70, 80, 90, 100, 110, 120, 20, 30, 130],
                    ),
                    column_info(
                        "Outcome2",
                        &[10],
                        UINT64,
                        &[400, 700, 800, 900, 1000, 1100, 1200, 200, 300, 1300],
                    ),
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                expected_info(vec![
                    (
                        NULL_HEADER,
                        &[1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                    ),
                    (
                        "Income1",
                        &[5, 0, 0, 0, 1, 2, 40, 70, 80, 90, 100, 110, 120, 0, 30, 130],
                    ),
                    (
                        "ID",
                        &[50, 0, 0, 0, 10, 20, 4, 7, 8, 9, 10, 11, 12, 0, 3, 13],
                    ),
                    (
                        "Outcome1",
                        &[500, 0, 0, 0, 100, 200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ),
                    (
                        "Outcome2",
                        &[
                            0, 0, 0, 0, 0, 0, 400, 700, 800, 900, 1000, 1100, 1200, 0, 300, 1300,
                        ],
                    ),
                ]),
            ),
            join_info(
                vec![
                    column_info(NULL_HEADER, &[1], BIT, &[1]),
                    column_info("ID", &[1], UINT64, &[5]),
                    column_info("Income1", &[1], UINT64, &[50]),
                    column_info("Outcome1", &[1], UINT64, &[500]),
                ],
                vec![
                    column_info(NULL_HEADER, &[1], BIT, &[1]),
                    column_info("ID", &[1], UINT64, &[5]),
                    column_info("Income2", &[1], UINT64, &[50]),
                    column_info("Outcome2", &[1], UINT64, &[51]),
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                expected_info(vec![
                    (NULL_HEADER, &[0, 1]),
                    ("ID", &[0, 5]),
                    ("Income1", &[0, 50]),
                    ("Outcome1", &[0, 500]),
                    ("Outcome2", &[0, 51]),
                ]),
            ),
            join_info(
                vec![
                    column_info(NULL_HEADER, &[1], BIT, &[1]),
                    column_info("Income1", &[1], UINT64, &[50]),
                    column_info("Outcome1", &[1], UINT64, &[500]),
                    column_info("ID", &[1], UINT64, &[5]),
                ],
                vec![
                    column_info(NULL_HEADER, &[1], BIT, &[1]),
                    column_info("ID", &[1], UINT64, &[5]),
                    column_info("Income2", &[1], UINT64, &[50]),
                    column_info("Outcome2", &[1], UINT64, &[51]),
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                expected_info(vec![
                    (NULL_HEADER, &[0, 1]),
                    ("Income1", &[0, 50]),
                    ("Outcome1", &[0, 500]),
                    ("ID", &[0, 5]),
                    ("Outcome2", &[0, 51]),
                ]),
            ),
            join_info(
                vec![
                    column_info(NULL_HEADER, &[2], BIT, &[1, 1]),
                    column_info("Income1", &[2], UINT64, &[40, 50]),
                    column_info("Outcome1", &[2], UINT64, &[400, 500]),
                    column_info("ID", &[2], UINT64, &[4, 5]),
                ],
                vec![
                    column_info(NULL_HEADER, &[2], BIT, &[1, 1]),
                    column_info("ID", &[2], UINT64, &[4, 3]),
                    column_info("Income2", &[2], UINT64, &[40, 30]),
                    column_info("Outcome2", &[2, 2], UINT64, &[40, 41, 30, 31]),
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                expected_info(vec![
                    (NULL_HEADER, &[0, 1, 1, 1]),
                    ("Income1", &[0, 50, 40, 30]),
                    ("Outcome1", &[0, 500, 400, 0]),
                    ("ID", &[0, 5, 4, 3]),
                    ("Outcome2", &[0, 0, 0, 0, 40, 41, 30, 31]),
                ]),
            ),
            join_info(
                vec![
                    column_info(NULL_HEADER, &[2], BIT, &[1, 1]),
                    column_info("Income1", &[2], UINT64, &[40, 50]),
                    column_info("Outcome1", &[2], UINT64, &[400, 500]),
                    column_info("ID", &[2], UINT64, &[4, 5]),
                ],
                vec![
                    column_info(NULL_HEADER, &[2], BIT, &[1, 1]),
                    column_info("ID", &[2], UINT64, &[6, 7]),
                    column_info("Income2", &[2], UINT64, &[60, 70]),
                    column_info("Outcome2", &[2, 2], UINT64, &[60, 61, 70, 71]),
                ],
                vec![("ID", "ID"), ("Income1", "Income2")],
                expected_info(vec![
                    (NULL_HEADER, &[1, 1, 1, 1]),
                    ("Income1", &[40, 50, 60, 70]),
                    ("Outcome1", &[400, 500, 0, 0]),
                    ("ID", &[4, 5, 6, 7]),
                    ("Outcome2", &[0, 0, 0, 0, 60, 61, 70, 71]),
                ]),
            ),
        ];
        join_helper(tests, JoinType::Full)?;

        Ok(())
    }

    fn gemm_helper(
        t0: Type,
        t1: Type,
        array0: Vec<u64>,
        array1: Vec<u64>,
        expected: Vec<u64>,
    ) -> Result<()> {
        let trans_perm0 = transpose_permutation(t0.get_shape().len());
        let trans_perm1 = transpose_permutation(t1.get_shape().len());

        let c = create_context()?;
        let g = c.create_graph()?;
        let i0 = g.input(t0.clone())?;
        let i1 = g.input(t1.clone())?;
        let trans_i0 = i0.permute_axes(trans_perm0)?;
        let trans_i1 = i1.permute_axes(trans_perm1)?;
        let gemm_false_false = i0.gemm(trans_i1.clone(), false, false)?;
        let gemm_false_true = i0.gemm(i1.clone(), false, true)?;
        let gemm_true_false = trans_i0.gemm(trans_i1, true, false)?;
        let gemm_true_true = trans_i0.gemm(i1, true, true)?;
        let o = g.create_tuple(vec![
            gemm_false_false.clone(),
            gemm_false_true,
            gemm_true_false,
            gemm_true_true,
        ])?;
        g.set_output_node(o.clone())?;
        g.finalize()?;
        c.set_main_graph(g.clone())?;
        c.finalize()?;

        let value0 = Value::from_flattened_array(&array0, t0.get_scalar_type())?;
        let value1 = Value::from_flattened_array(&array1, t1.get_scalar_type())?;
        let results = random_evaluate(g, vec![value0, value1])?.to_vector()?;

        let result_t = gemm_false_false.get_type()?;
        for result in results {
            assert_eq!(result.to_flattened_array_u64(result_t.clone())?, expected);
        }

        Ok(())
    }

    fn gemm_helper_random(t0: Type, t1: Type) -> Result<()> {
        let trans_perm0 = transpose_permutation(t0.get_shape().len());
        let trans_perm1 = transpose_permutation(t1.get_shape().len());

        let c = create_context()?;
        let g = c.create_graph()?;
        let i0 = g.random(t0.clone())?;
        let i1 = g.random(t1.clone())?;
        let trans_i0 = i0.permute_axes(trans_perm0)?;
        let trans_i1 = i1.permute_axes(trans_perm1)?;
        let gemm_false_false = i0.gemm(trans_i1.clone(), false, false)?;
        let gemm_false_true = i0.gemm(i1.clone(), false, true)?;
        let gemm_true_false = trans_i0.gemm(trans_i1.clone(), true, false)?;
        let gemm_true_true = trans_i0.gemm(i1, true, true)?;

        let mm = i0.matmul(trans_i1)?;
        let o = g.create_tuple(vec![
            mm.clone(),
            gemm_false_false,
            gemm_false_true,
            gemm_true_false,
            gemm_true_true,
        ])?;
        g.set_output_node(o.clone())?;
        g.finalize()?;
        c.set_main_graph(g.clone())?;
        c.finalize()?;

        let results = random_evaluate(g, vec![])?.to_vector()?;

        let result_t = mm.get_type()?;
        let expected = results[0].to_flattened_array_u64(result_t.clone())?;
        for result in results.iter().skip(1) {
            assert_eq!(result.to_flattened_array_u64(result_t.clone())?, expected);
        }

        Ok(())
    }

    #[test]
    fn test_gemm() {
        || -> Result<()> {
            gemm_helper(
                array_type(vec![2, 3], UINT32),
                array_type(vec![3, 3], UINT32),
                array!([[1, 2, 3], [4, 5, 6]]).into_raw_vec(),
                array!([[7, 8, 9], [10, 11, 12], [13, 14, 15]]).into_raw_vec(),
                array!([[50, 68, 86], [122, 167, 212]]).into_raw_vec(),
            )?;
            gemm_helper(
                array_type(vec![2, 2, 2], UINT32),
                array_type(vec![2, 3, 2], UINT32),
                vec![1, 2, 3, 4, 5, 6, 7, 8],
                vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
                vec![
                    50, 110, 170, 110, 250, 390, 830, 1050, 1270, 1130, 1430, 1730,
                ],
            )?;

            gemm_helper(
                array_type(vec![2, 127], BIT),
                array_type(vec![3, 127], BIT),
                vec![1; 2 * 127],
                vec![1; 3 * 127],
                vec![1, 1, 1, 1, 1, 1],
            )?;

            {
                let mut arr0 = vec![1; 127];
                arr0.extend(vec![0; 127]);

                let mut arr1 = vec![1; 127];
                arr1.extend(vec![0; 127]);
                arr1.extend(vec![1; 127]);

                gemm_helper(
                    array_type(vec![2, 127], BIT),
                    array_type(vec![3, 127], BIT),
                    arr0,
                    arr1,
                    vec![1, 0, 1, 0, 0, 0],
                )?;
            }
            gemm_helper(
                array_type(vec![2, 3], BIT),
                array_type(vec![3, 3], BIT),
                array!([[1, 0, 1], [0, 1, 1]]).into_raw_vec(),
                array!([[1, 1, 1], [0, 1, 0], [1, 1, 0]]).into_raw_vec(),
                vec![0, 0, 1, 0, 1, 1],
            )?;

            gemm_helper(
                array_type(vec![2, 9], BIT),
                array_type(vec![3, 9], BIT),
                array!([[1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 0, 0, 1, 0]]).into_raw_vec(),
                array!([
                    [1, 0, 1, 0, 1, 0, 1, 0, 1],
                    [0, 1, 0, 1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1, 0, 1, 0]
                ])
                .into_raw_vec(),
                vec![1, 0, 0, 0, 1, 1],
            )?;

            gemm_helper_random(array_type(vec![1, 1], BIT), array_type(vec![1, 1], BIT))?;
            gemm_helper_random(array_type(vec![127, 7], BIT), array_type(vec![127, 7], BIT))?;
            gemm_helper_random(
                array_type(vec![15, 7, 191], BIT),
                array_type(vec![15, 191], BIT),
            )?;
            gemm_helper_random(
                array_type(vec![15, 7, 191], BIT),
                array_type(vec![15, 15, 191], BIT),
            )?;

            Ok(())
        }()
        .unwrap();
    }

    fn concatenate_helper(
        input_types: Vec<Type>,
        axis: u64,
        result_type: Type,
        input_arrays: Vec<Vec<u64>>,
        expected: Vec<u64>,
    ) -> Result<()> {
        let c = simple_context(|g| {
            let mut inputs = vec![];
            for t in input_types.iter() {
                inputs.push(g.input((*t).clone())?);
            }
            g.concatenate(inputs, axis)
        })?;
        let g = c.get_main_graph()?;

        let mut input_values = vec![];
        for (i, t) in input_types.iter().enumerate() {
            input_values.push(Value::from_flattened_array(
                &input_arrays[i],
                t.get_scalar_type(),
            )?);
        }

        let result_value = evaluate_simple_evaluator(g, input_values, None)?;
        assert_eq!(result_value.to_flattened_array_u64(result_type)?, expected);
        Ok(())
    }

    #[test]
    fn test_concatenate() {
        || -> Result<()> {
            concatenate_helper(
                vec![
                    array_type(vec![1], INT32),
                    array_type(vec![1], INT32),
                    array_type(vec![1], INT32),
                ],
                0,
                array_type(vec![3], INT32),
                vec![vec![1], vec![2], vec![3]],
                vec![1, 2, 3],
            )?;
            concatenate_helper(
                vec![
                    array_type(vec![2, 1], UINT8),
                    array_type(vec![2, 2], UINT8),
                    array_type(vec![2, 3], UINT8),
                ],
                1,
                array_type(vec![2, 6], UINT8),
                vec![vec![1, 7], vec![2, 3, 8, 9], vec![4, 5, 6, 10, 11, 12]],
                (1..13).collect(),
            )?;
            concatenate_helper(
                vec![
                    array_type(vec![1, 2], UINT8),
                    array_type(vec![2, 2], UINT8),
                    array_type(vec![3, 2], UINT8),
                ],
                0,
                array_type(vec![6, 2], UINT8),
                vec![(1..3).collect(), (3..7).collect(), (7..13).collect()],
                (1..13).collect(),
            )?;
            concatenate_helper(
                vec![
                    array_type(vec![2, 1, 2], INT16),
                    array_type(vec![2, 2, 2], INT16),
                    array_type(vec![2, 3, 2], INT16),
                ],
                1,
                array_type(vec![2, 6, 2], INT16),
                vec![
                    vec![1, 2, 13, 14],
                    vec![3, 4, 5, 6, 15, 16, 17, 18],
                    vec![7, 8, 9, 10, 11, 12, 19, 20, 21, 22, 23, 24],
                ],
                (1..25).collect(),
            )?;
            concatenate_helper(
                vec![
                    array_type(vec![2, 1, 2], BIT),
                    array_type(vec![2, 2, 2], BIT),
                    array_type(vec![2, 3, 2], BIT),
                ],
                1,
                array_type(vec![2, 6, 2], BIT),
                vec![
                    vec![1, 0, 1, 1],
                    vec![0, 0, 0, 1, 1, 0, 0, 1],
                    vec![1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0],
                ],
                vec![
                    1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0,
                ],
            )?;

            Ok(())
        }()
        .unwrap();
    }

    fn print_helper(input: TypedValue) -> Result<()> {
        let c = create_context()?;
        let g = c.create_graph()?;
        let inp = g.input(input.t.clone())?;
        let o = g.print("Input".into(), inp)?;
        g.set_output_node(o)?;
        g.finalize()?;
        c.set_main_graph(g.clone())?;
        c.finalize()?;

        let result_value = evaluate_simple_evaluator(g, vec![input.value.clone()], None)?;
        assert_eq!(result_value, input.value);
        Ok(())
    }

    #[test]
    fn test_print() {
        || -> Result<()> {
            print_helper(TypedValue::from_scalar(42, INT32)?)?;
            print_helper(TypedValue::from_vector(vec![], FromVectorMode::Tuple)?)?;
            print_helper(TypedValue::from_ndarray(
                array![true, false, true].into_dyn(),
                BIT,
            )?)?;
            Ok(())
        }()
        .unwrap();
    }

    fn assert_helper(flag: TypedValue, input: TypedValue, expect_success: bool) -> Result<()> {
        let c = create_context()?;
        let g = c.create_graph()?;
        let inp0 = g.input(flag.t.clone())?;
        let inp1 = g.input(input.t.clone())?;
        let o = g.assert("Flag".into(), inp0, inp1)?;
        g.set_output_node(o)?;
        g.finalize()?;
        c.set_main_graph(g.clone())?;
        c.finalize()?;

        let result =
            evaluate_simple_evaluator(g, vec![flag.value.clone(), input.value.clone()], None);
        if expect_success {
            assert!(result.is_ok());
            assert_eq!(result?, input.value);
        } else {
            assert!(result.is_err());
        }
        Ok(())
    }

    #[test]
    fn test_assert() {
        || -> Result<()> {
            assert_helper(
                TypedValue::from_scalar(true, BIT)?,
                TypedValue::from_scalar(42, INT32)?,
                true,
            )?;
            assert_helper(
                TypedValue::from_scalar(false, BIT)?,
                TypedValue::from_vector(vec![], FromVectorMode::Tuple)?,
                false,
            )?;
            assert_helper(
                TypedValue::from_scalar(true, BIT)?,
                TypedValue::from_ndarray(array![true, false, true].into_dyn(), BIT)?,
                true,
            )?;
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_sort() -> Result<()> {
        let helper = |inputs: Vec<TypedValue>| -> Result<Vec<Vec<u64>>> {
            let c = simple_context(|g| {
                let nodes = inputs
                    .iter()
                    .map(|tv| g.input(tv.t.clone()))
                    .collect::<Result<Vec<_>>>()?;
                let key = "key".to_string();
                let mut named_nodes = vec![(
                    key.clone(),
                    nodes[0].a2b()?.get_slice(vec![
                        SliceElement::Ellipsis,
                        SliceElement::SubArray(None, None, Some(-1)),
                    ])?,
                )];
                for (i, node) in nodes.into_iter().enumerate().skip(1) {
                    named_nodes.push((format!("value_{}", i), node));
                }
                let result = g.sort(g.create_named_tuple(named_nodes)?, key.clone())?;
                let mut values = vec![result
                    .named_tuple_get(key)?
                    .get_slice(vec![
                        SliceElement::Ellipsis,
                        SliceElement::SubArray(None, None, Some(-1)),
                    ])?
                    .b2a(inputs[0].t.get_scalar_type())?];
                for i in (0..inputs.len()).into_iter().skip(1) {
                    values.push(result.named_tuple_get(format!("value_{}", i))?);
                }
                g.create_tuple(values)
            })?;
            let g = c.get_main_graph()?;
            let o = g.get_output_node()?;
            let result_value = random_evaluate(g, inputs.into_iter().map(|tv| tv.value).collect())?;
            let result_type = o.get_type()?;
            let types = match result_type {
                Type::Tuple(types) => types,
                _ => unreachable!(),
            };
            result_value
                .to_vector()?
                .into_iter()
                .zip(types.into_iter())
                .map(|(v, t)| v.to_flattened_array_u64((*t).clone()))
                .collect()
        };
        // Simple case.
        let input = Value::from_flattened_array(&[0, 1, 2, 2, 3], UINT32)?;
        let expected = vec![0, 1, 2, 2, 3];
        assert_eq!(
            helper(vec![TypedValue::new(array_type(vec![5], UINT32), input)?])?[0],
            expected
        );
        // Multiple arrays.
        let input1 = Value::from_flattened_array(&[3, 1, 2, 0, 2], UINT32)?;
        let input2 = Value::from_flattened_array(&[1, 2, 3, 44444444444444u64, 5], UINT64)?;
        let res = helper(vec![
            TypedValue::new(array_type(vec![5], UINT32), input1)?,
            TypedValue::new(array_type(vec![5], UINT64), input2)?,
        ])?;
        let expected = vec![vec![0, 1, 2, 2, 3], vec![44444444444444, 2, 3, 5, 1]];
        assert_eq!(res, expected);
        // Stable sort.
        let input1 = Value::from_flattened_array(&[3, 0, 0, 3, 0], UINT8)?;
        let input2 = Value::from_flattened_array(&[1, 2, 3, 4, 5], UINT64)?;
        let res = helper(vec![
            TypedValue::new(array_type(vec![5], UINT8), input1)?,
            TypedValue::new(array_type(vec![5], UINT64), input2)?,
        ])?;
        let expected = vec![vec![0, 0, 0, 3, 3], vec![2, 3, 5, 1, 4]];
        assert_eq!(res, expected);
        // 2d arrays.
        let input1 = Value::from_flattened_array(&[2, 1, 0], UINT8)?;
        let input2 = Value::from_flattened_array(&[1, 2, 3, 4, 5, 1, 2, 4, 5, 1, 1, 1], UINT64)?;
        let res = helper(vec![
            TypedValue::new(array_type(vec![3], UINT8), input1)?,
            TypedValue::new(array_type(vec![3, 2, 2], UINT64), input2)?,
        ])?;
        let expected = vec![vec![0, 1, 2], vec![5, 1, 1, 1, 5, 1, 2, 4, 1, 2, 3, 4]];
        assert_eq!(res, expected);

        // Incorrect sizes.
        let input1 = Value::from_flattened_array(&[1, 0, 0], UINT8)?;
        let input2 = Value::from_flattened_array(&[1, 2, 3, 4, 5], UINT64)?;
        assert!(helper(vec![
            TypedValue::new(array_type(vec![3], UINT8), input1)?,
            TypedValue::new(array_type(vec![5], UINT64), input2)?,
        ])
        .is_err());
        Ok(())
    }

    fn permutation_from_prf_helper(n: u64) -> Result<()> {
        let c = simple_context(|g| {
            let k = g.random(array_type(vec![128], BIT))?;
            g.permutation_from_prf(k, 0, n)
        })?;
        let g = c.get_main_graph()?;
        let o = g.get_output_node()?;
        let result_type = o.get_type()?;

        let mut evaluator = SimpleEvaluator {
            prng: PRNG::new(None)?,
            prfs: HashMap::new(),
        };

        let result_value = evaluator.evaluate_context(c.clone(), Vec::new())?;
        let perm = result_value.to_flattened_array_u64(result_type.clone())?;

        let mut perm_sorted = perm.clone();
        perm_sorted.sort();
        let range_vec: Vec<u64> = (0..n).collect();
        assert_eq!(perm_sorted, range_vec);

        Ok(())
    }

    #[test]
    fn test_permutation_from_prf() -> Result<()> {
        permutation_from_prf_helper(10)?;
        permutation_from_prf_helper(40)?;
        permutation_from_prf_helper(500)
    }

    fn value_to_flattened_array_u64(v: Value, t: Type) -> Result<Vec<u64>> {
        match t {
            Type::Scalar(st) => Ok(vec![v.to_u64(st)?]),
            Type::Array(_, _) => Ok(v.to_flattened_array_u64(t)?),
            Type::Tuple(vec_t) => Ok(v
                .to_vector()?
                .into_iter()
                .zip(vec_t.into_iter())
                .map(|(v, t)| value_to_flattened_array_u64(v, (*t).clone()))
                .collect::<Result<Vec<_>>>()?
                .concat()),
            _ => Err(runtime_error!("not implemented")),
        }
    }

    #[test]
    fn test_zeros() -> Result<()> {
        let helper = |t| {
            let c = simple_context(|g| g.zeros(t))?;
            let g = c.get_main_graph()?;
            let o = g.get_output_node()?;
            value_to_flattened_array_u64(random_evaluate(g, vec![])?, o.get_type()?)
        };
        assert_eq!(helper(scalar_type(INT32))?, vec![0]);
        assert_eq!(helper(array_type(vec![5], INT32))?, vec![0, 0, 0, 0, 0]);
        assert_eq!(helper(array_type(vec![3, 2], BIT))?, vec![0, 0, 0, 0, 0, 0]);
        assert_eq!(
            helper(tuple_type(vec![
                scalar_type(BIT),
                array_type(vec![2], INT32)
            ]))?,
            vec![0, 0, 0]
        );
        Ok(())
    }

    #[test]
    fn test_ones() -> Result<()> {
        let helper = |t| {
            let c = simple_context(|g| g.ones(t))?;
            let g = c.get_main_graph()?;
            let o = g.get_output_node()?;
            value_to_flattened_array_u64(random_evaluate(g, vec![])?, o.get_type()?)
        };
        assert_eq!(helper(scalar_type(INT32))?, vec![1]);
        assert_eq!(helper(array_type(vec![5], INT32))?, vec![1, 1, 1, 1, 1]);
        assert_eq!(helper(array_type(vec![3, 2], BIT))?, vec![1, 1, 1, 1, 1, 1]);
        assert_eq!(
            helper(tuple_type(vec![
                scalar_type(BIT),
                array_type(vec![2], INT32)
            ]))?,
            vec![1, 1, 1]
        );
        Ok(())
    }

    #[test]
    fn test_cum_sum() -> Result<()> {
        let cum_sum = |tv: TypedValue, axis: u64| -> Result<TypedValue> {
            let c = simple_context(|g| g.cum_sum(g.input(tv.t)?, axis))?;
            let g = c.get_main_graph()?;
            let o = g.get_output_node()?;
            TypedValue::new(o.get_type()?, random_evaluate(g, vec![tv.value])?)
        };
        let tv = |arr, st| TypedValue::from_ndarray(arr, st);
        assert_eq!(
            cum_sum(tv(array![1, 1, 1, 1, 1].into_dyn(), UINT8)?, 0)?,
            tv(array![1, 2, 3, 4, 5].into_dyn(), UINT8)?
        );
        assert_eq!(
            cum_sum(tv(array![[1, 2, 3], [4, 5, 6]].into_dyn(), INT32)?, 0)?,
            tv(array![[1, 2, 3], [5, 7, 9]].into_dyn(), INT32)?
        );
        assert_eq!(
            cum_sum(tv(array![[1, 2, 3], [4, 5, 6]].into_dyn(), INT64)?, 1)?,
            tv(array![[1, 3, 6], [4, 9, 15]].into_dyn(), INT64)?
        );
        assert_eq!(
            cum_sum(tv(array![1, 1, 1, 1, 1].into_dyn(), BIT)?, 0)?,
            tv(array![1, 0, 1, 0, 1].into_dyn(), BIT)?
        );
        Ok(())
    }
}
