use crate::broadcast::{index_to_number, number_to_index};
use crate::bytes::{
    add_u64, add_vectors_u64, multiply_u64, multiply_vectors_u64, subtract_vectors_u64,
};
use crate::bytes::{vec_from_bytes, vec_to_bytes};
use crate::data_types::{array_type, Type, BIT};
use crate::data_values::Value;
use crate::errors::Result;
use crate::evaluators::Evaluator;
use crate::graphs::{Node, Operation};
use crate::random::{Prf, PRNG, SEED_SIZE};
use crate::slices::slice_index;

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::iter::repeat;

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
                let result_shape = result_type.get_shape();
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
                let values = dependencies_values[0].to_flattened_array_u64(t.clone())?;
                let cur_shape = t.get_shape();
                let res_shape = node.get_type()?.get_shape();
                let mut result = vec![0u64; values.len()];
                for i in 0..values.len() as u64 {
                    let old_index = number_to_index(i, &cur_shape);
                    let mut new_index = vec![];
                    for j in perm.iter() {
                        new_index.push(old_index[*j as usize]);
                    }
                    result[index_to_number(&new_index, &res_shape) as usize] = values[i as usize];
                }
                Value::from_flattened_array(&result, t.get_scalar_type())
            }
            Operation::Sum(axes) => {
                let dependency = node.get_node_dependencies()[0].clone();
                let inp_t = dependency.get_type()?;
                let values = dependencies_values[0].to_flattened_array_u64(inp_t.clone())?;
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
                            Ok(dependencies_values[0].clone())
                        } else {
                            let inp_shape = inp_t.get_shape();
                            let res_len: u64 = res_shape.iter().product();
                            let mut result = vec![0u64; res_len as usize];
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
            Operation::Reshape(new_type) => {
                let dependency_value = dependencies_values[0].clone();
                let dependency_value_flattened = flatten_value(dependency_value);
                let new_value = unflatten_value(&dependency_value_flattened, &mut 0, new_type);
                Ok(new_value)
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
            Operation::Random(t) => {
                let new_value = self.prng.get_random_value(t)?;
                Ok(new_value)
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
            _ => Err(runtime_error!("Not implemented")),
        }
    }
}

#[cfg(tests)]
mod tests {
    #[test]
    fn test_prf() {
        let helper = |iv: u64, t: Type| -> Result<()> {
            let c = create_context()?;
            let g = c.create_graph()?;
            let i1 = g.random(array_type(vec![128], BIT))?;
            let i2 = g.random(array_type(vec![128], BIT))?;
            let p1 = g.prf(i1.clone(), iv, t.clone())?;
            let p2 = g.prf(i2, iv, t.clone())?;
            let p3 = g.prf(i1, iv, t.clone())?;
            let o = g.create_vector(t.clone(), vec![p1, p2, p3])?;
            g.set_output_node(o)?;
            g.finalize()?;
            c.set_main_graph(g)?;
            c.finalize()?;
            let mut evaluator = SimpleEvaluator {
                prng: PRNG::new(None)?,
                prfs: HashMap::new(),
            };
            let v = evaluator.evaluate_context(c, Vec::new())?;
            let ot = vector_type(3, t.clone());
            assert_eq!(evaluator.prfs.len(), 2);
            assert!((*v).borrow().check_type(ot)?);
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
}
