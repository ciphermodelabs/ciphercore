use crate::broadcast::{broadcast_arrays, broadcast_shapes};
use crate::custom_ops::Instantiation;
use crate::data_types::{
    array_type, get_named_types, is_valid_shape, named_tuple_type, scalar_size_in_bits,
    scalar_type, tuple_type, vector_type, ArrayShape, ScalarType, Type, BIT, UINT32, UINT64,
};
use crate::errors::Result;
use crate::graphs::{create_context, Context, JoinType, Node, Operation, WeakContext};
use crate::slices::get_slice_shape;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

type CachedResults = HashMap<(u64, u64), Type>;
type CachedInstantiations = HashMap<Instantiation, Type>;

fn get_node_global_id(node: Node) -> (u64, u64) {
    let graph_id = node.get_graph().get_id();
    let node_id = node.get_id();
    (graph_id, node_id)
}

#[doc(hidden)]
pub struct TypeInferenceWorker {
    context: WeakContext,
    cached_results: CachedResults,
    cached_instantiations: CachedInstantiations,
}

#[doc(hidden)]
pub fn create_type_inference_worker(context: Context) -> TypeInferenceWorker {
    TypeInferenceWorker {
        context: context.downgrade(),
        cached_results: CachedResults::new(),
        cached_instantiations: CachedInstantiations::new(),
    }
}

fn mixed_multiply_inference(t0: Type, t1: Type) -> Result<Type> {
    if !t0.is_scalar() && !t0.is_array() {
        return Err(runtime_error!(
            "The first argument of mixed multiply is not a scalar or an array: {t0:?}"
        ));
    }
    if !t1.is_scalar() && !t1.is_array() {
        return Err(runtime_error!(
            "The second argument of mixed multiply is not a scalar or an array: {t1:?}"
        ));
    }

    if t0.get_scalar_type() == BIT {
        return Err(runtime_error!(
            "The scalar type of the first argument shouldn't be BIT: {t0:?}"
        ));
    }
    if t1.get_scalar_type() != BIT {
        return Err(runtime_error!(
            "The scalar type of the second argument must be BIT: {t1:?}"
        ));
    }

    if t1.is_scalar() {
        return Ok(t0);
    }
    if t0.is_scalar() {
        return Ok(array_type(t1.get_shape(), t0.get_scalar_type()));
    }

    Ok(array_type(
        broadcast_shapes(t0.get_shape(), t1.get_shape())?,
        t0.get_scalar_type(),
    ))
}

fn dot_type_inference(t0: Type, t1: Type) -> Result<Type> {
    if !t0.is_scalar() && !t0.is_array() {
        return Err(runtime_error!(
            "The first argument of dot is not a scalar or an array: {t0:?}"
        ));
    }
    if !t1.is_scalar() && !t1.is_array() {
        return Err(runtime_error!(
            "The second argument of dot is not a scalar or an array: {t1:?}"
        ));
    }
    if t0.get_scalar_type() != t1.get_scalar_type() {
        return Err(runtime_error!(
            "Incompatible scalar types: {t0:?} vs {t1:?}"
        ));
    }
    let st = t0.get_scalar_type();
    if t0.is_array() && t1.is_array() {
        let s0 = t0.get_shape();
        let s1 = t1.get_shape();
        if s0.len() == 1 && s1.len() == 1 {
            if s0[0] != s1[0] {
                return Err(runtime_error!(
                    "Dot with incompatible dimensions: {s0:?} vs {s1:?}"
                ));
            }
            Ok(scalar_type(st))
        } else if s1.len() == 1 {
            if s0[s0.len() - 1] != s1[0] {
                Err(runtime_error!(
                    "Dot with incompatible dimensions: {s0:?} vs {s1:?}"
                ))
            } else {
                let mut sr = s0.clone();
                sr.remove(s0.len() - 1);
                Ok(array_type(sr, st))
            }
        } else if s0[s0.len() - 1] != s1[s1.len() - 2] {
            Err(runtime_error!(
                "Dot with incompatible dimensions: {s0:?} vs {s1:?}"
            ))
        } else {
            let mut sr = s0.clone();
            sr.remove(s0.len() - 1);
            for i in 0..s1.len() {
                if i != s1.len() - 2 {
                    sr.push(s1[i]);
                }
            }
            Ok(array_type(sr, st))
        }
    } else if t0.is_array() {
        Ok(t0)
    } else {
        Ok(t1)
    }
}

fn matmul_type_inference(t0: Type, t1: Type) -> Result<Type> {
    if !t0.is_array() {
        return Err(runtime_error!(
            "The first argument of matmul is not an array: {t0:?}"
        ));
    }
    if !t1.is_array() {
        return Err(runtime_error!(
            "The second argument of matmul is not an array: {t1:?}"
        ));
    }
    if t0.get_scalar_type() != t1.get_scalar_type() {
        return Err(runtime_error!(
            "Incompatible scalar types: {t0:?} vs {t1:?}"
        ));
    }
    let st = t0.get_scalar_type();
    let mut s0 = t0.get_shape();
    let mut s1 = t1.get_shape();
    let remove_dim0 = s0.len() == 1;
    if s0.len() == 1 {
        s0.insert(0, 1);
    }
    let remove_dim1 = s1.len() == 1;
    if s1.len() == 1 {
        s1.push(1);
    }
    if s0[s0.len() - 1] != s1[s1.len() - 2] {
        return Err(runtime_error!(
            "Matmul with incompatible dimensions: {s0:?} vs {s1:?}"
        ));
    }
    let mut result_dims =
        broadcast_shapes(s0[0..s0.len() - 2].to_vec(), s1[0..s1.len() - 2].to_vec())?;
    if !remove_dim0 {
        result_dims.push(s0[s0.len() - 2]);
    }
    if !remove_dim1 {
        result_dims.push(s1[s1.len() - 1]);
    }
    if result_dims.is_empty() {
        Ok(scalar_type(st))
    } else {
        Ok(array_type(result_dims, st))
    }
}

pub(crate) fn transpose_shape(shape: ArrayShape, transpose_flag: bool) -> ArrayShape {
    let num_dims = shape.len();
    if transpose_flag && num_dims > 1 {
        let mut res = shape;
        res.swap(num_dims - 2, num_dims - 1);
        res
    } else {
        shape
    }
}

fn gemm_type_inference(t0: Type, t1: Type, transpose0: bool, transpose1: bool) -> Result<Type> {
    if !t0.is_array() {
        return Err(runtime_error!(
            "The first argument of gemm is not an array: {t0:?}"
        ));
    }
    if !t1.is_array() {
        return Err(runtime_error!(
            "The second argument of gemm is not an array: {t1:?}"
        ));
    }
    if t0.get_scalar_type() != t1.get_scalar_type() {
        return Err(runtime_error!(
            "Incompatible scalar types: {t0:?} vs {t1:?}"
        ));
    }
    let input_shape0 = t0.get_shape();
    let input_shape1 = t1.get_shape();
    if input_shape0.len() == 1 || input_shape1.len() == 1 {
        return Err(runtime_error!(
            "To multiply vectors, use matmul or dot operations instead of gemm. Shapes are: {input_shape0:?} and {input_shape1:?}."
        ));
    }

    let st = t0.get_scalar_type();
    let s0 = transpose_shape(input_shape0, transpose0);
    let s1 = transpose_shape(input_shape1, transpose1);

    if s0[s0.len() - 1] != s1[s1.len() - 2] {
        return Err(runtime_error!(
            "Gemm with incompatible dimensions: {s0:?} vs {s1:?}"
        ));
    }
    let mut result_dims =
        broadcast_shapes(s0[0..s0.len() - 2].to_vec(), s1[0..s1.len() - 2].to_vec())?;
    result_dims.push(s0[s0.len() - 2]);
    result_dims.push(s1[s1.len() - 1]);

    Ok(array_type(result_dims, st))
}

pub(super) fn a2b_type_inference(original_type: Type) -> Result<Type> {
    if !original_type.is_scalar() && !original_type.is_array() {
        return Err(runtime_error!(
            "Invalid type for A2B: can only be array or scalar: {original_type:?}"
        ));
    }
    let st = original_type.get_scalar_type();
    if st == BIT {
        return Err(runtime_error!(
            "A2B can't be applied to bits, got {original_type:?}"
        ));
    }
    let sz = scalar_size_in_bits(st);
    if original_type.is_scalar() {
        Ok(array_type(vec![sz], BIT))
    } else {
        let mut shape = original_type.get_shape();
        shape.push(sz);
        Ok(array_type(shape, BIT))
    }
}

fn b2a_type_inference(t: Type, st: ScalarType) -> Result<Type> {
    if !t.is_valid() {
        return Err(runtime_error!("Invalid type: {t:?}"));
    }
    if !t.is_array() {
        return Err(runtime_error!("Trying to B2A non-array: {t:?}"));
    }
    let mut shape = t.get_shape();
    let array_st = t.get_scalar_type();
    if array_st != BIT {
        return Err(runtime_error!("Trying to B2A from non-bits: {t:?}"));
    }
    if st == BIT {
        return Err(runtime_error!("Trying to B2A into bits: {t:?}"));
    }
    if shape[shape.len() - 1] != scalar_size_in_bits(st.clone()) {
        return Err(runtime_error!("Invalid scalar type for B2A: {t:?}"));
    }
    if shape.len() == 1 {
        Ok(scalar_type(st))
    } else {
        shape.pop();
        Ok(array_type(shape, st))
    }
}

/// Name of the "null" column that contains bits indicating whether the corresponding row is void of content.
/// If the "null" bit is zero, the row is empty.
pub const NULL_HEADER: &str = "row_mask_sentinel_639bcf36-a1b0-11ed-b93a-423c7c497182";

fn check_table_and_extract_column_types(
    t: Type,
    has_null_column: bool,
) -> Result<(HashMap<String, Arc<Type>>, u64)> {
    let v = get_named_types(&t)?;

    if has_null_column && v.len() < 2 {
        return Err(runtime_error!("Named tuple should contain at least two columns, one of which must be the null column. Got: {v:?}"));
    }
    if !has_null_column && v.is_empty() {
        return Err(runtime_error!(
            "Named tuple should contain at least one column."
        ));
    }
    let mut num_entries = 0;
    let mut contains_null = false;
    let mut all_headers: HashMap<String, Arc<Type>> = HashMap::new();
    for (h, sub_t) in v {
        if !sub_t.is_array() {
            return Err(runtime_error!(
                "Named tuple should consist of arrays, got: {sub_t:?}"
            ));
        }
        let shape = sub_t.get_shape();
        if num_entries == 0 {
            num_entries = shape[0]
        }
        if num_entries != shape[0] {
            return Err(runtime_error!(
                "Number of entries should be the same in each column: {} vs {}",
                num_entries,
                shape[0]
            ));
        }
        if h == NULL_HEADER && has_null_column {
            if sub_t.get_scalar_type() != BIT {
                return Err(runtime_error!(
                    "Null column should be binary, got {sub_t:?}"
                ));
            }
            if shape != vec![num_entries] {
                return Err(runtime_error!(
                    "Null column should have shape {:?}",
                    vec![num_entries]
                ));
            }
            contains_null = true;
        }
        all_headers.insert(h.clone(), (*sub_t).clone());
    }
    if !contains_null && has_null_column {
        return Err(runtime_error!("Named tuple should contain the null column"));
    }
    Ok((all_headers, num_entries))
}

fn join_inference(
    t0: Type,
    t1: Type,
    join_t: JoinType,
    headers: HashMap<String, String>,
) -> Result<Type> {
    if headers.is_empty() {
        return Err(runtime_error!("No column headers provided"));
    }

    let (headers_types_map0, num_entries0) =
        check_table_and_extract_column_types(t0.clone(), true)?;
    let (headers_types_map1, num_entries1) =
        check_table_and_extract_column_types(t1.clone(), true)?;

    let mut key_headers1 = vec![];
    for (h0, h1) in headers {
        if h0 == NULL_HEADER || h1 == NULL_HEADER {
            return Err(runtime_error!("Join along the null column is forbidden"));
        }
        if !headers_types_map0.contains_key(&h0) {
            return Err(runtime_error!(
                "There is no header {h0} in the first named tuple"
            ));
        }
        if !headers_types_map1.contains_key(&h1) {
            return Err(runtime_error!(
                "There is no header {h1} in the second named tuple"
            ));
        }
        let sub_t0 = headers_types_map0.get(&h0).unwrap();
        let sub_t1 = headers_types_map1.get(&h1).unwrap();

        let shape0 = sub_t0.get_shape();
        let shape1 = sub_t1.get_shape();
        // First dimension can differ as input tuples might have different number of entries/rows
        if shape0[1..] != shape1[1..] {
            return Err(runtime_error!(
                "Columns with names {h0} {h1} have incompatible shapes to be compared"
            ));
        }
        if sub_t0.get_scalar_type() != sub_t1.get_scalar_type() {
            return Err(runtime_error!(
                "Columns with names {h0} {h1} have incompatible scalar types to be compared"
            ));
        }
        key_headers1.push(h1);
    }
    for (h, _) in headers_types_map1 {
        if h != NULL_HEADER && headers_types_map0.contains_key(&h) && !key_headers1.contains(&h) {
            return Err(runtime_error!("Both tuples contain columns named {h} that don't participate in the join. Rename one of these to a unique name."));
        }
    }

    let mut result_types_vec = vec![];
    let res_num_entries = match join_t {
        JoinType::Inner | JoinType::Left => num_entries0,
        JoinType::Union | JoinType::Full => num_entries0 + num_entries1,
    };

    let headers_types0 = get_named_types(&t0)?;
    let headers_types1 = get_named_types(&t1)?;

    for (h, sub_t) in headers_types0 {
        let mut shape = sub_t.get_shape();
        shape[0] = res_num_entries;
        let st = sub_t.get_scalar_type();
        let res_sub_t = array_type(shape, st);
        result_types_vec.push((h.clone(), res_sub_t));
    }

    for (h, sub_t) in headers_types1 {
        if !headers_types_map0.contains_key(h) && !key_headers1.contains(h) {
            let mut shape = sub_t.get_shape();
            shape[0] = res_num_entries;
            let st = sub_t.get_scalar_type();
            let res_sub_t = array_type(shape, st);
            result_types_vec.push((h.clone(), res_sub_t));
        }
    }

    Ok(named_tuple_type(result_types_vec))
}

/// Returns Some(n) if a given operation requires n node dependencies.
/// None means the number can be variable.
fn get_number_of_node_dependencies(operation: Operation) -> Option<u64> {
    match operation {
        Operation::Input(_)
        | Operation::Random(_)
        | Operation::Constant(_, _)
        | Operation::RandomPermutation(_) => Some(0),
        Operation::Truncate(_)
        | Operation::Sum(_)
        | Operation::PermuteAxes(_)
        | Operation::InversePermutation
        | Operation::CuckooToPermutation
        | Operation::Get(_)
        | Operation::GetSlice(_)
        | Operation::Reshape(_)
        | Operation::NOP
        | Operation::PRF(_, _)
        | Operation::A2B
        | Operation::B2A(_)
        | Operation::TupleGet(_)
        | Operation::NamedTupleGet(_)
        | Operation::Repeat(_)
        | Operation::ArrayToVector
        | Operation::VectorToArray
        | Operation::DecomposeSwitchingMap(_)
        | Operation::Print(_)
        | Operation::Shard(_) => Some(1),
        Operation::Add
        | Operation::Subtract
        | Operation::Multiply
        | Operation::MixedMultiply
        | Operation::Dot
        | Operation::Matmul
        | Operation::VectorGet
        | Operation::Gather(_)
        | Operation::Iterate
        | Operation::CuckooHash
        | Operation::Join(_, _)
        | Operation::Gemm(_, _)
        | Operation::Assert(_) => Some(2),
        Operation::SegmentCumSum => Some(3),
        Operation::Stack(_)
        | Operation::Concatenate(_)
        | Operation::CreateTuple
        | Operation::CreateNamedTuple(_)
        | Operation::CreateVector(_)
        | Operation::Zip
        | Operation::Call
        | Operation::Custom(_) => None,
    }
}

/// Returns Some(n) if a given operation requires n graph dependencies.
/// None means the number can be variable.
fn get_number_of_graph_dependencies(operation: Operation) -> Option<u64> {
    match operation {
        Operation::Call | Operation::Iterate => Some(1),
        _ => Some(0),
    }
}

fn flatten_type_size(t: Type) -> Result<u64> {
    let err = || runtime_error!("Overflow during flatten_type_size()");
    match t {
        Type::Scalar(_) | Type::Array(_, _) => Ok(1),
        Type::Tuple(v) => {
            let mut result: u64 = 0;
            for x in v {
                result = result
                    .checked_add(flatten_type_size((*x).clone())?)
                    .ok_or_else(err)?;
            }
            Ok(result)
        }
        Type::NamedTuple(v) => {
            let mut result: u64 = 0;
            for (_, x) in v {
                result = result
                    .checked_add(flatten_type_size((*x).clone())?)
                    .ok_or_else(err)?;
            }
            Ok(result)
        }
        Type::Vector(len, t1) => {
            if len == 0 {
                return Ok(0);
            }
            Ok(flatten_type_size((*t1).clone())?
                .checked_mul(len)
                .ok_or_else(err)?)
        }
    }
}

fn flatten_type(t: Type) -> Vec<Type> {
    match t {
        Type::Scalar(_) | Type::Array(_, _) => {
            vec![t]
        }
        Type::Tuple(v) => {
            let mut result = vec![];
            for x in v {
                for y in flatten_type((*x).clone()) {
                    result.push(y);
                }
            }
            result
        }
        Type::NamedTuple(v) => {
            let mut result = vec![];
            for (_, x) in v {
                for y in flatten_type((*x).clone()) {
                    result.push(y);
                }
            }
            result
        }
        Type::Vector(len, t1) => {
            let mut result = vec![];
            for _ in 0..len {
                for x in flatten_type((*t1).clone()) {
                    result.push(x);
                }
            }
            result
        }
    }
}

/// Panics when t1 or t2 are not scalar or array
fn can_atomic_reshape(t1: Type, t2: Type) -> bool {
    if !t1.is_scalar() && !t1.is_array() {
        panic!("can_atomic_reshape with invalid arguments");
    }
    if !t2.is_scalar() && !t2.is_array() {
        panic!("can_atomic_reshape with invalid arguments");
    }
    if t1.get_scalar_type() != t2.get_scalar_type() {
        return false;
    }
    let shape1 = if t1.is_scalar() {
        vec![]
    } else {
        t1.get_shape()
    };
    let shape2 = if t2.is_scalar() {
        vec![]
    } else {
        t2.get_shape()
    };
    if !shape1.is_empty() && !is_valid_shape(shape1.clone()) {
        return false;
    }
    if !shape2.is_empty() && !is_valid_shape(shape2.clone()) {
        return false;
    }
    let v1: u64 = shape1.iter().product();
    let v2: u64 = shape2.iter().product();
    v1 == v2
}

impl TypeInferenceWorker {
    fn register_result(&mut self, node: Node, result: Type) -> Result<()> {
        if !result.is_valid() {
            return Err(runtime_error!("Trying to register invalid type"));
        }
        self.cached_results.insert(get_node_global_id(node), result);
        Ok(())
    }

    pub(super) fn unregister_node(&mut self, node: Node) -> Result<()> {
        self.cached_results.remove(&get_node_global_id(node));
        Ok(())
    }

    pub fn cached_node_type(&self, node: &Node) -> Result<Option<Type>> {
        if self.context.upgrade() != node.get_graph().get_context() {
            return Err(runtime_error!(
                "Can't process a node from a different context"
            ));
        }

        let node_global_id = get_node_global_id(node.clone());
        Ok(self.cached_results.get(&node_global_id).cloned())
    }

    pub fn process_node(&mut self, node: Node) -> Result<Type> {
        let cached_result = self.cached_node_type(&node)?;
        if let Some(result) = cached_result {
            return Ok(result);
        }

        let node_dependencies = node.get_node_dependencies();
        let number_of_node_dependencies = get_number_of_node_dependencies(node.get_operation());
        if let Some(n) = number_of_node_dependencies {
            if node_dependencies.len() as u64 != n {
                return Err(runtime_error!("Invalid number of node dependencies"));
            }
        }
        let mut node_dependencies_types = vec![];
        for dependency in &node_dependencies {
            node_dependencies_types.push(self.process_node(dependency.clone())?);
        }
        let graph_dependencies = node.get_graph_dependencies();
        let number_of_graph_dependencies = get_number_of_graph_dependencies(node.get_operation());
        if let Some(n) = number_of_graph_dependencies {
            if graph_dependencies.len() as u64 != n {
                return Err(runtime_error!("Invalid number of graph dependencies"));
            }
        }
        match node.get_operation() {
            Operation::Input(input_type) => {
                if !input_type.is_valid() {
                    return Err(runtime_error!("Input with an invalid type: {input_type:?}"));
                }
                let result = input_type;
                self.register_result(node, result.clone())?;
                Ok(result)
            }
            Operation::Add | Operation::Subtract | Operation::Multiply => {
                let result = broadcast_arrays(vec![
                    node_dependencies_types[0].clone(),
                    node_dependencies_types[1].clone(),
                ])?;
                self.register_result(node, result.clone())?;
                Ok(result)
            }
            Operation::MixedMultiply => {
                let result = mixed_multiply_inference(
                    node_dependencies_types[0].clone(),
                    node_dependencies_types[1].clone(),
                )?;
                self.register_result(node, result.clone())?;
                Ok(result)
            }
            Operation::Dot => {
                let result = dot_type_inference(
                    node_dependencies_types[0].clone(),
                    node_dependencies_types[1].clone(),
                )?;
                self.register_result(node, result.clone())?;
                Ok(result)
            }
            Operation::Matmul => {
                let result = matmul_type_inference(
                    node_dependencies_types[0].clone(),
                    node_dependencies_types[1].clone(),
                )?;
                self.register_result(node, result.clone())?;
                Ok(result)
            }
            Operation::Gemm(transpose0, transpose1) => {
                let result = gemm_type_inference(
                    node_dependencies_types[0].clone(),
                    node_dependencies_types[1].clone(),
                    transpose0,
                    transpose1,
                )?;
                self.register_result(node, result.clone())?;
                Ok(result)
            }
            Operation::Join(join_t, headers) => {
                let result = join_inference(
                    node_dependencies_types[0].clone(),
                    node_dependencies_types[1].clone(),
                    join_t,
                    headers,
                )?;
                self.register_result(node, result.clone())?;
                Ok(result)
            }
            Operation::Truncate(d) => {
                let t = node_dependencies_types[0].clone();
                if d == 0 {
                    return Err(runtime_error!("Can't divide by zero"));
                }
                if !t.is_array() && !t.is_scalar() {
                    return Err(runtime_error!("Can't truncate this type: {t:?}"));
                }
                if t.get_scalar_type().get_signed() && d > i64::MAX as u64 {
                    return Err(runtime_error!("Scale for truncation is too large: {d}"));
                }
                self.register_result(node, t.clone())?;
                Ok(t)
            }
            Operation::Sum(s) => {
                let t = node_dependencies_types[0].clone();
                if !t.is_array() {
                    return Err(runtime_error!("Can't sum this type: {t:?}"));
                }
                let os = t.get_shape();
                let mut tmp = s.clone();
                tmp.sort_unstable();
                tmp.dedup();
                if tmp.len() < s.len() {
                    return Err(runtime_error!("Non-unique axes: {s:?}"));
                }
                let mut set: HashSet<u64> = HashSet::new();
                for x in s {
                    if x >= os.len() as u64 {
                        return Err(runtime_error!("Invalid axis: {x}"));
                    }
                    set.insert(x);
                }
                let mut rs = vec![];
                for (i, item) in os.iter().enumerate() {
                    if !set.contains(&(i as u64)) {
                        rs.push(*item);
                    }
                }
                let st = t.get_scalar_type();
                let result = if rs.is_empty() {
                    scalar_type(st)
                } else {
                    array_type(rs, st)
                };
                self.register_result(node, result.clone())?;
                Ok(result)
            }
            Operation::PermuteAxes(s) => {
                let t = node_dependencies_types[0].clone();
                if !t.is_array() {
                    return Err(runtime_error!("Can't permute_axes this type: {t:?}"));
                }
                let os = t.get_shape();
                let mut tmp = s.clone();
                tmp.sort_unstable();
                tmp.dedup();
                if tmp.len() < s.len() {
                    return Err(runtime_error!("Non-unique axes: {s:?}"));
                }
                for x in &s {
                    if *x >= os.len() as u64 {
                        return Err(runtime_error!("Invalid axes: {s:?}"));
                    }
                }
                if s.len() != os.len() {
                    return Err(runtime_error!("Not a permutation: {s:?}"));
                }
                let mut rs = vec![];
                for i in 0..s.len() {
                    rs.push(os[s[i] as usize]);
                }
                let st = t.get_scalar_type();
                let result = array_type(rs, st);
                self.register_result(node, result.clone())?;
                Ok(result)
            }
            Operation::InversePermutation => {
                let t = node_dependencies_types[0].clone();
                if !t.is_array() {
                    return Err(runtime_error!("Input type should be an array: {t:?}"));
                }
                if t.get_scalar_type() != UINT64 {
                    return Err(runtime_error!(
                        "Input elements must be unsigned 64-bit integers: {t:?}"
                    ));
                }
                if t.get_shape().len() > 1 {
                    return Err(runtime_error!(
                        "Input type should be an array with one dimension: {:?}",
                        t.get_shape()
                    ));
                }
                self.register_result(node, t.clone())?;
                Ok(t)
            }
            Operation::CuckooToPermutation => {
                let t = node_dependencies_types[0].clone();
                if !t.is_array() {
                    return Err(runtime_error!("Input type should be an array: {t:?}"));
                }
                if t.get_scalar_type() != UINT64 {
                    return Err(runtime_error!(
                        "Input elements must be 64-bit integers: {t:?}"
                    ));
                }
                self.register_result(node, t.clone())?;
                Ok(t)
            }
            Operation::DecomposeSwitchingMap(n) => {
                let t = node_dependencies_types[0].clone();
                if !t.is_array() {
                    return Err(runtime_error!("Input type should be an array: {t:?}"));
                }
                if t.get_scalar_type() != UINT64 {
                    return Err(runtime_error!(
                        "Input elements must be 64-bit integers: {t:?}"
                    ));
                }
                let shape = t.get_shape();
                if shape[0] > n {
                    return Err(runtime_error!(
                        "Switching map is longer than expected: {shape:?} vs {n:?}"
                    ));
                }
                let duplication_map_t = tuple_type(vec![
                    array_type(shape.clone(), UINT64),
                    array_type(shape, BIT),
                ]);
                let output_t = tuple_type(vec![t.clone(), duplication_map_t, t]);
                Ok(output_t)
            }
            Operation::Get(s) => {
                let t = node_dependencies_types[0].clone();
                if !t.is_array() {
                    return Err(runtime_error!("Can't run get on this type: {t:?}"));
                }
                let os = t.get_shape();
                if s.len() > os.len() {
                    return Err(runtime_error!("Too long index: {s:?}"));
                }
                for i in 0..s.len() {
                    if s[i] >= os[i] {
                        return Err(runtime_error!("Out of bounds: {s:?}"));
                    }
                }
                let st = t.get_scalar_type();
                if s.len() == os.len() {
                    let result = scalar_type(st);
                    self.register_result(node, result.clone())?;
                    return Ok(result);
                }
                let mut rs = vec![];
                for item in os.iter().skip(s.len()) {
                    rs.push(*item);
                }
                let result = array_type(rs, st);
                self.register_result(node, result.clone())?;
                Ok(result)
            }
            Operation::GetSlice(slice) => {
                let t = node_dependencies_types[0].clone();
                if !t.is_array() {
                    return Err(runtime_error!("Can't run get_slice on this type: {t:?}"));
                }
                let os = t.get_shape();
                let st = t.get_scalar_type();
                let ns = get_slice_shape(os, slice)?;
                let result = if ns.is_empty() {
                    scalar_type(st)
                } else {
                    array_type(ns, st)
                };
                self.register_result(node, result.clone())?;
                Ok(result)
            }
            Operation::Reshape(new_type) => {
                let old_type = node_dependencies_types[0].clone();
                if flatten_type_size(old_type.clone())? != flatten_type_size(new_type.clone())? {
                    return Err(runtime_error!(
                        "Incompatible types for reshape: {old_type:?} to {new_type:?}"
                    ));
                }
                let v1 = flatten_type(old_type);
                let v2 = flatten_type(new_type.clone());
                for i in 0..v1.len() {
                    if !can_atomic_reshape(v1[i].clone(), v2[i].clone()) {
                        return Err(runtime_error!(
                            "Incompatible types for reshape: {:?} to {:?}",
                            v1[i],
                            v2[i]
                        ));
                    }
                }
                let result = new_type;
                self.register_result(node, result.clone())?;
                Ok(result)
            }
            Operation::NOP => {
                let t = node_dependencies_types[0].clone();
                self.register_result(node, t.clone())?;
                Ok(t)
            }
            Operation::Random(t) => {
                self.register_result(node, t.clone())?;
                Ok(t)
            }
            Operation::RandomPermutation(n) => {
                if n == 0 {
                    return Err(runtime_error!("Permutation length should be non-zero"));
                }
                let t = array_type(vec![n], UINT64);
                self.register_result(node, t.clone())?;
                Ok(t)
            }
            Operation::PRF(_, ot) => {
                let t = node_dependencies_types[0].clone();
                if !t.is_array() {
                    return Err(runtime_error!("PRF key must be an array: {t:?}"));
                }
                let s = t.get_shape();
                let st = t.get_scalar_type();
                if s.len() != 1 || s[0] != 128 || st != BIT {
                    return Err(runtime_error!("PRF key must consist of 128 bits: {s:?}"));
                }
                self.register_result(node, ot.clone())?;
                Ok(ot)
            }
            Operation::Stack(outer_shape) => {
                if !is_valid_shape(outer_shape.clone()) {
                    return Err(runtime_error!("Invalid outer shape: {outer_shape:?}"));
                }
                let mut pr = 1;
                for x in &outer_shape {
                    pr *= *x;
                }
                if node_dependencies.len() as u64 != pr {
                    return Err(runtime_error!(
                        "Stack with a wrong number of arguments: {:?} vs {:?}",
                        node_dependencies.len(),
                        pr
                    ));
                }
                let inner_type = broadcast_arrays(node_dependencies_types)?;
                let st = inner_type.get_scalar_type();
                let result = if inner_type.is_scalar() {
                    array_type(outer_shape, st)
                } else {
                    let mut rs = outer_shape;
                    for x in &inner_type.get_shape() {
                        rs.push(*x);
                    }
                    array_type(rs, st)
                };
                self.register_result(node, result.clone())?;
                Ok(result)
            }
            Operation::Concatenate(axis) => {
                if node_dependencies.len() < 2 {
                    return Err(runtime_error!(
                        "Concatenate should have at least two input arrays, got {:?}",
                        node_dependencies.len()
                    ));
                }
                for t in &node_dependencies_types {
                    if !t.is_array() {
                        return Err(runtime_error!(
                            "All inputs of Concatenate must be arrays, got {t:?}"
                        ));
                    }
                }
                let first_type = &node_dependencies_types[0];
                let st = first_type.get_scalar_type();

                let mut result_shape = first_type.get_shape();
                if axis >= result_shape.len() as u64 {
                    return Err(runtime_error!("Wrong concatenation axis: {axis:?}"));
                }
                for t in node_dependencies_types.iter().skip(1) {
                    if t.get_scalar_type() != st {
                        return Err(runtime_error!(
                            "Inputs have different scalar types: {:?} vs {:?}",
                            st,
                            t.get_scalar_type()
                        ));
                    }
                    let shape = t.get_shape();
                    if result_shape.len() != shape.len() {
                        return Err(runtime_error!(
                            "Inputs have shapes of different length: {:?} vs {:?}",
                            result_shape.len(),
                            shape.len()
                        ));
                    }
                    for (i, d) in shape.iter().enumerate() {
                        if result_shape[i] != *d && axis != i as u64 {
                            return Err(runtime_error!(
                                "Inputs have incompatible shapes: {result_shape:?} vs {shape:?}"
                            ));
                        }
                    }
                    result_shape[axis as usize] += shape[axis as usize];
                }
                let result = array_type(result_shape, st);
                Ok(result)
            }
            Operation::Constant(t, ref value) => {
                if !value.check_type(t.clone())? {
                    return Err(runtime_error!(
                        "Invalid constant type: {t:?} for value {value:?}"
                    ));
                }
                self.register_result(node, t.clone())?;
                Ok(t)
            }
            Operation::A2B => {
                let original_type = node_dependencies_types[0].clone();
                let result = a2b_type_inference(original_type)?;
                self.register_result(node, result.clone())?;
                Ok(result)
            }
            Operation::B2A(scalar_type) => {
                let original_type = node_dependencies_types[0].clone();
                let result = b2a_type_inference(original_type, scalar_type)?;
                self.register_result(node, result.clone())?;
                Ok(result)
            }
            Operation::CreateTuple => {
                let mut types = vec![];
                for dependency_type in node_dependencies_types {
                    types.push(dependency_type);
                }
                let result = tuple_type(types);
                self.register_result(node, result.clone())?;
                Ok(result)
            }
            Operation::CreateNamedTuple(fields) => {
                if node_dependencies.len() != fields.len() {
                    return Err(runtime_error!(
                        "Invalid number of fields provided: {:?} vs {:?}",
                        node_dependencies.len(),
                        fields.len()
                    ));
                }
                let mut types = vec![];
                for dependency_type in node_dependencies_types {
                    types.push(dependency_type);
                }
                let result = named_tuple_type(
                    fields
                        .iter()
                        .zip(types.iter())
                        .map(|(x, y)| (x.clone(), y.clone()))
                        .collect(),
                );
                self.register_result(node, result.clone())?;
                Ok(result)
            }
            Operation::CreateVector(element_type) => {
                for dependency_type in node_dependencies_types.clone() {
                    if dependency_type != element_type {
                        return Err(runtime_error!(
                            "Vector element type mismatch: {dependency_type:?} vs {element_type:?}"
                        ));
                    }
                }
                let result = vector_type(node_dependencies_types.len() as u64, element_type);
                self.register_result(node, result.clone())?;
                Ok(result)
            }
            Operation::TupleGet(field_id) => {
                let original_type = node_dependencies_types[0].clone();
                let result = match original_type {
                    Type::Tuple(fields) => {
                        if field_id >= fields.len() as u64 {
                            return Err(runtime_error!(
                                "Index is out of bounds: {:?} vs {:?}",
                                field_id,
                                fields.len()
                            ));
                        }
                        (*fields[field_id as usize]).clone()
                    }
                    Type::NamedTuple(fields) => {
                        if field_id >= fields.len() as u64 {
                            return Err(runtime_error!(
                                "Index is out of bounds: {:?} vs {:?}",
                                field_id,
                                fields.len()
                            ));
                        }
                        let (_, t) = fields[field_id as usize].clone();
                        (*t).clone()
                    }
                    _ => {
                        return Err(runtime_error!(
                            "Can't TupleGet from this type: {original_type:?}"
                        ));
                    }
                };
                self.register_result(node, result.clone())?;
                Ok(result)
            }
            Operation::NamedTupleGet(field_name) => {
                let original_type = node_dependencies_types[0].clone();
                match original_type {
                    Type::NamedTuple(fields) => {
                        for (s, t) in &fields {
                            if *s == field_name {
                                let result = (**t).clone();
                                self.register_result(node, result.clone())?;
                                return Ok(result);
                            }
                        }
                    }
                    _ => {
                        return Err(runtime_error!(
                            "Can't NamedTupleGet from this type: {original_type:?}"
                        ));
                    }
                };
                Err(runtime_error!("Invalid field name: {field_name:?}"))
            }
            Operation::VectorGet => {
                let index_type = node_dependencies_types[1].clone();
                if index_type != scalar_type(UINT64) && index_type != scalar_type(UINT32) {
                    return Err(runtime_error!(
                        "Vector index must be an UINT64 or UINT32, got {index_type:?}"
                    ));
                }
                let vector_type = node_dependencies_types[0].clone();
                let result = match vector_type {
                    Type::Vector(_, inner_type) => (*inner_type).clone(),
                    _ => {
                        return Err(runtime_error!(
                            "VectorGet can only be applied to vectors: {vector_type:?}"
                        ));
                    }
                };
                self.register_result(node, result.clone())?;
                Ok(result)
            }
            Operation::Zip => {
                if node_dependencies.len() < 2 {
                    return Err(runtime_error!(
                        "Zip with a wrong number of arguments: {:?}",
                        node_dependencies.len()
                    ));
                }
                let mut types = vec![];
                for dependency_type in node_dependencies_types {
                    types.push(dependency_type);
                }
                let mut length: Option<u64> = None;
                let mut element_types = vec![];
                for t in &types {
                    match t {
                        Type::Vector(l, et) => {
                            match length {
                                Some(len) => {
                                    if *l != len {
                                        return Err(runtime_error!(
                                            "Zip of uneven lengths: {len:?} vs {l:?}"
                                        ));
                                    }
                                }
                                None => {
                                    length = Some(*l);
                                }
                            }
                            element_types.push((*et).clone());
                        }
                        _ => {
                            return Err(runtime_error!(
                                "An argument of zip is not a vector: {t:?}"
                            ));
                        }
                    }
                }
                match length {
                    Some(len) => {
                        let result = vector_type(len, Type::Tuple(element_types));
                        self.register_result(node, result.clone())?;
                        Ok(result)
                    }
                    None => {
                        panic!("Should not be here!");
                    }
                }
            }
            Operation::Repeat(n) => {
                let other_type = node_dependencies_types[0].clone();
                let result = vector_type(n, other_type);
                self.register_result(node, result.clone())?;
                Ok(result)
            }
            Operation::Call => {
                let graph = graph_dependencies[0].clone();
                graph.check_finalized()?;
                let mut input_types = vec![];
                for graph_node in &graph.get_nodes() {
                    if let Operation::Input(t) = graph_node.get_operation() {
                        input_types.push(t.clone());
                    }
                }
                if node_dependencies.len() != input_types.len() {
                    return Err(runtime_error!(
                        "Invalid number of arguments in Call: {:?} vs {:?}",
                        node_dependencies.len(),
                        input_types.len()
                    ));
                }
                for i in 0..node_dependencies.len() {
                    if input_types[i] != node_dependencies_types[i] {
                        return Err(runtime_error!(
                            "Type mismatch for argument {}: expected {:?}, got {:?}",
                            i,
                            input_types[i],
                            node_dependencies_types[i]
                        ));
                    }
                }
                let result = self.process_node(graph.get_output_node()?)?;
                self.register_result(node, result.clone())?;
                Ok(result)
            }
            Operation::Iterate => {
                let graph = graph_dependencies[0].clone();
                graph.check_finalized()?;
                let mut input_types = vec![];
                for graph_node in &graph.get_nodes() {
                    if let Operation::Input(t) = graph_node.get_operation() {
                        input_types.push(t.clone());
                    }
                }
                if input_types.len() != 2 {
                    return Err(runtime_error!(
                        "Iterate graph must have two inputs: state and current input. Got {} inputs", input_types.len()
                    ));
                }
                let state_type = input_types[0].clone();
                let input_sequence_type = input_types[1].clone();
                let output_type = self.process_node(graph.get_output_node()?)?;
                match output_type {
                    Type::Tuple(element_types) => {
                        if element_types.len() != 2 {
                            return Err(runtime_error!(
                                "Iterate graph must output a tuple of two elements as an output, got {} elements", element_types.len()
                            ));
                        }
                        if *element_types[0] != state_type {
                            return Err(runtime_error!(
                                "State type mismatch: expected {:?}, got {:?}",
                                state_type,
                                element_types[0]
                            ));
                        }
                        let output_sequence_type = (*element_types[1]).clone();
                        let t0 = node_dependencies_types[0].clone();
                        let t1 = node_dependencies_types[1].clone();
                        if t0 != state_type {
                            return Err(runtime_error!(
                                "Invalid state type: expected {state_type:?}, got {t0:?}"
                            ));
                        }
                        match t1 {
                            Type::Vector(len, element_type) => {
                                if *element_type != input_sequence_type {
                                    return Err(runtime_error!(
                                        "Invalid sequence type: expected {input_sequence_type:?}, got {element_type:?}"
                                    ));
                                }
                                let result = tuple_type(vec![
                                    state_type,
                                    vector_type(len, output_sequence_type),
                                ]);
                                self.register_result(node, result.clone())?;
                                Ok(result)
                            }
                            _ => Err(runtime_error!(
                                "Invalid sequence type: expected vector, got {t1:?}"
                            )),
                        }
                    }
                    _ => Err(runtime_error!(
                        "Iterate graph must output a tuple: got {output_type:?}"
                    )),
                }
            }
            Operation::ArrayToVector => {
                let t = node_dependencies_types[0].clone();
                if !t.is_array() {
                    return Err(runtime_error!(
                        "ArrayToVector applied to a non-array: {t:?}"
                    ));
                }
                let st = t.get_scalar_type();
                let shape = t.get_shape();
                let result = if shape.len() == 1 {
                    vector_type(shape[0], scalar_type(st))
                } else {
                    let mut cut_shape = shape.clone();
                    cut_shape.drain(0..1);
                    vector_type(shape[0], array_type(cut_shape, st))
                };
                self.register_result(node, result.clone())?;
                Ok(result)
            }
            Operation::VectorToArray => {
                let t = node_dependencies_types[0].clone();
                if let Type::Vector(length, element_type) = t {
                    if length == 0 {
                        return Err(runtime_error!(
                            "VectorToArray can't be applied to an empty vector"
                        ));
                    }
                    if !element_type.is_scalar() && !element_type.is_array() {
                        return Err(runtime_error!(
                            "VectorToArray can be only applied to a vector of scalars or arrays, got {element_type:?}"
                        ));
                    }
                    let st = element_type.get_scalar_type();
                    let result = if element_type.is_scalar() {
                        array_type(vec![length], st)
                    } else {
                        let mut shape = element_type.get_shape();
                        let mut modified_shape = vec![length];
                        modified_shape.append(&mut shape);
                        array_type(modified_shape, st)
                    };
                    self.register_result(node, result.clone())?;
                    Ok(result)
                } else {
                    Err(runtime_error!(
                        "VectorToArray can't be applied to a non-vector: {t:?}"
                    ))
                }
            }
            Operation::Gather(axis) => {
                let input_t = node_dependencies_types[0].clone();
                if !input_t.is_array() {
                    return Err(runtime_error!(
                        "Take can be only applied to an array: {input_t:?}"
                    ));
                }
                let indices_t = node_dependencies_types[1].clone();
                // TODO: support UINT32
                if !matches!(indices_t, Type::Array(_, UINT64)) {
                    return Err(runtime_error!(
                        "Indices must be an array of UINT64: {indices_t:?}"
                    ));
                }
                let input_shape = input_t.get_shape();
                if axis >= input_shape.len() as u64 {
                    return Err(runtime_error!(
                        "Invalid axis. The axis index should be smaller than {}, got {}",
                        input_shape.len(),
                        axis
                    ));
                }
                let indices_shape = indices_t.get_shape();
                let indices_size = indices_shape.iter().product::<u64>();
                if indices_size > input_shape[axis as usize] {
                    return Err(runtime_error!(
                        "Number of indices is too big: {}. At most {} elements can be extracted.",
                        indices_size,
                        input_shape[axis as usize]
                    ));
                }
                let mut result_shape = input_shape[0..axis as usize].to_vec();
                result_shape.extend_from_slice(&indices_shape);
                result_shape.extend_from_slice(&input_shape[(axis + 1) as usize..]);
                let result = array_type(result_shape, input_t.get_scalar_type());
                self.register_result(node, result.clone())?;
                Ok(result)
            }
            Operation::CuckooHash => {
                let input_t = node_dependencies_types[0].clone();
                let hash_t = node_dependencies_types[1].clone();
                if !matches!(input_t, Type::Array(_, BIT)) {
                    return Err(runtime_error!(
                        "CuckooHash can't be applied to a non-binary arrays: {input_t:?}"
                    ));
                }
                let input_shape = input_t.get_shape();
                if input_shape.len() < 2 {
                    return Err(runtime_error!(
                        "Input shape must have at least 2 dimensions, got {input_shape:?}"
                    ));
                }
                if !matches!(hash_t, Type::Array(_, BIT)) {
                    return Err(runtime_error!(
                        "CuckooHash needs a binary array as a hash matrix: {hash_t:?}"
                    ));
                }
                let hash_shape = hash_t.get_shape();
                if hash_shape.len() != 3 {
                    return Err(runtime_error!(
                        "Hash array should have 3 dimensions: {hash_shape:?}"
                    ));
                }
                if hash_shape[0] < 3 {
                    return Err(runtime_error!(
                        "At least 3 hash matrices should be provided, got {}",
                        hash_shape[0]
                    ));
                }
                if hash_shape[1] > 63 {
                    return Err(runtime_error!(
                        "Hash map is too big: {}. Decrease the number of rows of hash matrices",
                        hash_shape[1]
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
                // For each subarray, the output hash map contains indices of this array
                let mut output_shape = input_shape[0..input_shape.len() - 2].to_vec();
                let hash_map_size = 1 << hash_shape[1];
                output_shape.push(hash_map_size);
                let result = array_type(output_shape, UINT64);
                self.register_result(node, result.clone())?;
                Ok(result)
            }
            Operation::SegmentCumSum => {
                let input_t = node_dependencies_types[0].clone();
                let binary_t = node_dependencies_types[1].clone();
                let first_t = node_dependencies_types[2].clone();

                if !input_t.is_array() {
                    return Err(runtime_error!(
                        "First argument must be an array: {input_t:?}"
                    ));
                }
                let input_shape = input_t.get_shape();
                if Type::Array(vec![input_shape[0]], BIT) != binary_t {
                    return Err(runtime_error!(
                        "Second argument must be a one-dimensional binary array of length {}",
                        input_shape[0]
                    ));
                }
                let input_st = input_t.get_scalar_type();
                if input_shape.len() == 1 {
                    if Type::Scalar(input_st) != first_t {
                        return Err(runtime_error!(
                            "Input array and first row have different scalar types: {input_t:?} and {first_t:?}"
                        ));
                    }
                } else if Type::Array(input_shape[1..].to_vec(), input_st) != first_t {
                    return Err(runtime_error!(
                        "Input array and first row are incompatible: {input_t:?} and {first_t:?}"
                    ));
                }

                let mut result_shape = input_shape;
                result_shape[0] += 1;
                let result_t = array_type(result_shape, input_t.get_scalar_type());

                self.register_result(node, result_t.clone())?;
                Ok(result_t)
            }
            Operation::Shard(shard_config) => {
                let input_t = node_dependencies_types[0].clone();
                let (headers_types, num_entries) =
                    check_table_and_extract_column_types(input_t.clone(), false)?;
                let headers: Vec<String> = headers_types.keys().cloned().collect();

                if shard_config.num_shards > 700 {
                    return Err(runtime_error!(
                        "No more than 700 shards can be handled, {} provided",
                        shard_config.num_shards
                    ));
                }

                let num_elements_in_all_shards = shard_config.shard_size * shard_config.num_shards;
                if num_entries > num_elements_in_all_shards {
                    return Err(runtime_error!("Input elements can't fit given shards. Shards can contain {} elements, while input has {}", num_elements_in_all_shards, num_entries));
                }
                if shard_config.shard_headers.is_empty() {
                    return Err(runtime_error!(
                        "At least one shard header should be provided"
                    ));
                }
                let mut shard_headers_dedup = shard_config.shard_headers.clone();
                shard_headers_dedup.sort_unstable();
                shard_headers_dedup.dedup();
                if shard_headers_dedup.len() != shard_config.shard_headers.len() {
                    return Err(runtime_error!("Sharding headers contain duplicates"));
                }
                for h in &shard_config.shard_headers {
                    if !headers.contains(h) {
                        return Err(runtime_error!("Sharding can't be done along the column {}.  There is no such input column.", h));
                    }
                }

                let shard_mask_t = array_type(vec![shard_config.shard_size], BIT);

                let mut shard_data_t_vec = vec![];
                let headers_types = get_named_types(&input_t)?;
                for (h, sub_t) in headers_types {
                    let mut shape = sub_t.get_shape();
                    shape[0] = shard_config.shard_size;
                    let st = sub_t.get_scalar_type();
                    let res_sub_t = array_type(shape, st);
                    shard_data_t_vec.push((h.clone(), res_sub_t));
                }
                let shard_data_t = named_tuple_type(shard_data_t_vec);

                let mut result_tuple_vector = vec![];
                for _ in 0..shard_config.num_shards {
                    result_tuple_vector
                        .push(tuple_type(vec![shard_mask_t.clone(), shard_data_t.clone()]));
                }
                let result_t = tuple_type(result_tuple_vector);
                self.register_result(node, result_t.clone())?;
                Ok(result_t)
            }
            Operation::Print(_) => {
                let input_t = node_dependencies_types[0].clone();
                self.register_result(node, input_t.clone())?;
                Ok(input_t)
            }
            Operation::Assert(_) => {
                let condition_t = node_dependencies_types[0].clone();
                if !condition_t.is_scalar() || condition_t.get_scalar_type() != BIT {
                    return Err(runtime_error!(
                        "Assertion condition must be a scalar bit: {condition_t:?}"
                    ));
                }
                let input_t = node_dependencies_types[1].clone();
                self.register_result(node, input_t.clone())?;
                Ok(input_t)
            }
            // Here we can end up in an infinite loop due
            // to circular dependencies between instantiations.
            // For now we just crash with stack overflow.
            // In the future, we will need some safeguard here.
            Operation::Custom(op) => {
                let instantiation = Instantiation {
                    op: op.clone(),
                    arguments_types: node_dependencies_types.clone(),
                };
                let maybe_result = self.cached_instantiations.get(&instantiation);
                if let Some(t) = maybe_result {
                    let t = t.clone();
                    self.register_result(node, t.clone())?;
                    return Ok(t);
                }
                let fake_context = create_context()?;
                let instantiated_graph =
                    op.instantiate(fake_context.clone(), node_dependencies_types)?;
                let result = instantiated_graph.get_output_node()?.get_type()?;
                instantiated_graph.set_as_main()?;
                fake_context.finalize()?;
                self.register_result(node, result.clone())?;
                Ok(result)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_types::{
        create_scalar_type, ArrayShape, Type, BIT, INT32, INT8, UINT32, UINT8,
    };
    use crate::data_values::Value;
    use crate::graphs::{
        create_unchecked_context, Graph, JoinType, ShardConfig, Slice, SliceElement,
    };

    #[test]
    fn test_malformed() {
        let context1 = create_unchecked_context().unwrap();
        let context2 = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context1.clone());
        let graph = context2.create_graph().unwrap();
        let t = scalar_type(BIT);
        let node = graph.input(t.clone()).unwrap();
        let e = worker.process_node(node);
        assert!(e.is_err());
    }
    #[test]
    fn test_input() {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let t = scalar_type(BIT);
        let node = graph.input(t.clone()).unwrap();
        assert_eq!(worker.process_node(node).unwrap(), t.clone());
    }

    #[test]
    fn test_add() {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i1 = graph.input(array_type(vec![2, 1, 5], BIT)).unwrap();
        let i2 = graph.input(array_type(vec![7, 1], BIT)).unwrap();
        let o = graph.add(i1.clone(), i2.clone()).unwrap();
        let t = worker.process_node(o.clone()).unwrap();
        assert_eq!(t, array_type(vec![2, 7, 5], BIT));
    }

    fn mixed_multiply_helper(t0: Type, t1: Type, expected: Option<Type>) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i1 = graph.input(t0).unwrap();
        let i2 = graph.input(t1).unwrap();
        let o = graph.mixed_multiply(i1.clone(), i2.clone()).unwrap();
        let t = worker.process_node(o.clone());
        if let Some(expected_t) = expected {
            assert_eq!(t.unwrap(), expected_t);
        } else {
            assert!(t.is_err());
        }
    }

    #[test]
    fn test_mixed_multiply() {
        mixed_multiply_helper(
            array_type(vec![2, 1, 5], INT32),
            array_type(vec![7, 1], BIT),
            Some(array_type(vec![2, 7, 5], INT32)),
        );
        mixed_multiply_helper(
            scalar_type(INT32),
            array_type(vec![7, 1], BIT),
            Some(array_type(vec![7, 1], INT32)),
        );
        mixed_multiply_helper(
            array_type(vec![2, 1, 5], INT32),
            scalar_type(BIT),
            Some(array_type(vec![2, 1, 5], INT32)),
        );

        // malformed
        mixed_multiply_helper(
            array_type(vec![2, 1, 5], BIT),
            array_type(vec![7, 1], BIT),
            None,
        );
        mixed_multiply_helper(
            array_type(vec![2, 1, 5], INT32),
            array_type(vec![7, 1], INT32),
            None,
        );
        mixed_multiply_helper(
            vector_type(5, scalar_type(INT32)),
            array_type(vec![7, 1], BIT),
            None,
        );
        mixed_multiply_helper(
            array_type(vec![7, 1], INT32),
            vector_type(5, scalar_type(BIT)),
            None,
        );
    }

    fn test_dot_worker(t0: Type, t1: Type, t2: Type) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i0 = graph.input(t0).unwrap();
        let i1 = graph.input(t1).unwrap();
        let out = graph.dot(i0, i1).unwrap();
        let t2_result = worker.process_node(out).unwrap();
        assert_eq!(t2_result, t2);
    }

    fn test_dot_worker_fail(t0: Type, t1: Type) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i0 = graph.input(t0).unwrap();
        let i1 = graph.input(t1).unwrap();
        let out = graph.dot(i0, i1).unwrap();
        let e = worker.process_node(out);
        assert!(e.is_err());
    }

    #[test]
    fn test_dot() {
        test_dot_worker(
            array_type(vec![10], INT32),
            array_type(vec![10], INT32),
            scalar_type(INT32),
        );
        test_dot_worker(
            array_type(vec![10, 20], INT32),
            array_type(vec![20, 30], INT32),
            array_type(vec![10, 30], INT32),
        );
        test_dot_worker(
            array_type(vec![10, 20], INT32),
            scalar_type(INT32),
            array_type(vec![10, 20], INT32),
        );
        test_dot_worker(
            scalar_type(INT32),
            array_type(vec![10, 20], INT32),
            array_type(vec![10, 20], INT32),
        );
        test_dot_worker(scalar_type(INT32), scalar_type(INT32), scalar_type(INT32));
        test_dot_worker(
            array_type(vec![10, 20, 50], INT32),
            array_type(vec![50], INT32),
            array_type(vec![10, 20], INT32),
        );
        test_dot_worker(
            array_type(vec![10, 20, 50], INT32),
            array_type(vec![30, 50, 70], INT32),
            array_type(vec![10, 20, 30, 70], INT32),
        );
        test_dot_worker_fail(scalar_type(INT32), scalar_type(BIT));
        test_dot_worker_fail(tuple_type(vec![]), scalar_type(BIT));
        test_dot_worker_fail(array_type(vec![50], INT32), array_type(vec![100], INT32));
        test_dot_worker_fail(
            array_type(vec![10, 50], INT32),
            array_type(vec![100, 3], INT32),
        );
        test_dot_worker_fail(
            array_type(vec![10, 50], INT32),
            array_type(vec![100], INT32),
        );
        test_dot_worker_fail(
            array_type(vec![10, 50, 100], INT32),
            array_type(vec![100, 30, 50], INT32),
        );
    }

    fn test_matmul_worker(t0: Type, t1: Type, t2: Type) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i0 = graph.input(t0).unwrap();
        let i1 = graph.input(t1).unwrap();
        let out = graph.matmul(i0, i1).unwrap();
        let t2_result = worker.process_node(out).unwrap();
        assert_eq!(t2_result, t2);
    }

    fn test_matmul_worker_fail(t0: Type, t1: Type) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i0 = graph.input(t0).unwrap();
        let i1 = graph.input(t1).unwrap();
        let out = graph.matmul(i0, i1).unwrap();
        let e = worker.process_node(out);
        assert!(e.is_err());
    }

    #[test]
    fn test_matmul() {
        test_matmul_worker(
            array_type(vec![10], INT32),
            array_type(vec![10], INT32),
            scalar_type(INT32),
        );
        test_matmul_worker(
            array_type(vec![10, 20], INT32),
            array_type(vec![20, 30], INT32),
            array_type(vec![10, 30], INT32),
        );
        test_matmul_worker(
            array_type(vec![10, 20, 50], INT32),
            array_type(vec![50], INT32),
            array_type(vec![10, 20], INT32),
        );
        test_matmul_worker(
            array_type(vec![10, 20, 50], INT32),
            array_type(vec![10, 50, 70], INT32),
            array_type(vec![10, 20, 70], INT32),
        );
        test_matmul_worker(
            array_type(vec![10, 20, 30], INT32),
            array_type(vec![30, 70], INT32),
            array_type(vec![10, 20, 70], INT32),
        );
        test_matmul_worker(
            array_type(vec![20, 30], INT32),
            array_type(vec![10, 30, 70], INT32),
            array_type(vec![10, 20, 70], INT32),
        );
        test_matmul_worker(
            array_type(vec![1, 3, 4, 20, 30], INT32),
            array_type(vec![5, 3, 1, 30, 70], INT32),
            array_type(vec![5, 3, 4, 20, 70], INT32),
        );
        test_matmul_worker_fail(scalar_type(INT32), scalar_type(BIT));
        test_matmul_worker_fail(tuple_type(vec![]), scalar_type(BIT));
        test_matmul_worker_fail(array_type(vec![50], INT32), array_type(vec![100], INT32));
        test_matmul_worker_fail(
            array_type(vec![10, 50], INT32),
            array_type(vec![100, 3], INT32),
        );
        test_matmul_worker_fail(
            array_type(vec![10, 50], INT32),
            array_type(vec![100], INT32),
        );
        test_matmul_worker_fail(
            array_type(vec![10, 50, 100], INT32),
            array_type(vec![100, 30, 50], INT32),
        );
        test_matmul_worker_fail(
            array_type(vec![10, 20, 50], INT32),
            array_type(vec![30, 50, 70], INT32),
        );
        test_matmul_worker_fail(array_type(vec![10, 20], INT32), scalar_type(INT32));
        test_matmul_worker_fail(scalar_type(INT32), array_type(vec![10, 20], INT32));
        test_matmul_worker_fail(scalar_type(INT32), scalar_type(INT32));
    }

    fn test_truncate_worker(t: Type, scale: u64) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i = graph.input(t.clone()).unwrap();
        let out = graph.truncate(i, scale).unwrap();
        let t_result = worker.process_node(out).unwrap();
        assert_eq!(t_result, t);
    }

    fn test_truncate_worker_fail(t: Type, scale: u64) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i = graph.input(t.clone()).unwrap();
        let out = graph.truncate(i, scale).unwrap();
        let e = worker.process_node(out);
        assert!(e.is_err());
    }

    #[test]
    fn test_truncate() {
        test_truncate_worker(array_type(vec![10, 20], INT32), 1000);
        test_truncate_worker(scalar_type(INT32), 1000);
        test_truncate_worker(scalar_type(INT32), i64::MAX as u64);
        test_truncate_worker(scalar_type(UINT64), 1u64 << 32);
        test_truncate_worker_fail(array_type(vec![10, 20], INT32), 0);
        test_truncate_worker_fail(scalar_type(INT32), 0);
        test_truncate_worker_fail(tuple_type(vec![]), 1000);
        test_truncate_worker_fail(scalar_type(INT32), i64::MAX as u64 + 1);
    }

    fn test_sum_worker(t0: Type, s: ArrayShape, t1: Type) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i = graph.input(t0.clone()).unwrap();
        let out = graph.sum(i, s).unwrap();
        let t_result = worker.process_node(out).unwrap();
        assert_eq!(t_result, t1);
    }

    fn test_sum_worker_fail(t0: Type, s: ArrayShape) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i = graph.input(t0.clone()).unwrap();
        let out = graph.sum(i, s).unwrap();
        let e = worker.process_node(out);
        assert!(e.is_err());
    }

    #[test]
    fn test_sum() {
        test_sum_worker(
            array_type(vec![10, 20, 30], INT32),
            vec![],
            array_type(vec![10, 20, 30], INT32),
        );
        test_sum_worker(
            array_type(vec![10, 20, 30], INT32),
            vec![0],
            array_type(vec![20, 30], INT32),
        );
        test_sum_worker(
            array_type(vec![10, 20, 30], INT32),
            vec![1, 0],
            array_type(vec![30], INT32),
        );
        test_sum_worker(
            array_type(vec![10, 20, 30], INT32),
            vec![1, 0, 2],
            scalar_type(INT32),
        );
        test_sum_worker_fail(array_type(vec![10, 20, 30], INT32), vec![0, 0]);
        test_sum_worker_fail(array_type(vec![10, 20, 30], INT32), vec![3, 1]);
        test_sum_worker_fail(scalar_type(INT32), vec![]);
        test_sum_worker_fail(tuple_type(vec![]), vec![]);
    }

    fn test_permute_axes_worker(t0: Type, s: ArrayShape, t1: Type) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i = graph.input(t0.clone()).unwrap();
        let out = graph.permute_axes(i, s).unwrap();
        let t_result = worker.process_node(out).unwrap();
        assert_eq!(t_result, t1);
    }

    fn test_permute_axes_worker_fail(t0: Type, s: ArrayShape) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i = graph.input(t0.clone()).unwrap();
        let out = graph.permute_axes(i, s).unwrap();
        let e = worker.process_node(out);
        assert!(e.is_err());
    }

    #[test]
    fn test_permute_axes() {
        test_permute_axes_worker(
            array_type(vec![10, 20, 30], INT32),
            vec![1, 2, 0],
            array_type(vec![20, 30, 10], INT32),
        );
        test_permute_axes_worker(
            array_type(vec![10, 20, 30], INT32),
            vec![2, 1, 0],
            array_type(vec![30, 20, 10], INT32),
        );
        test_permute_axes_worker(
            array_type(vec![10, 20, 30], BIT),
            vec![0, 1, 2],
            array_type(vec![10, 20, 30], BIT),
        );
        test_permute_axes_worker_fail(array_type(vec![10, 20, 30], BIT), vec![0, 0, 2]);
        test_permute_axes_worker_fail(array_type(vec![10, 20, 30], BIT), vec![0, 1, 3]);
        test_permute_axes_worker_fail(array_type(vec![10, 20, 30], BIT), vec![1, 0]);
        test_permute_axes_worker_fail(scalar_type(BIT), vec![]);
        test_permute_axes_worker_fail(tuple_type(vec![]), vec![]);
    }

    fn test_get_worker(t0: Type, s: ArrayShape, t1: Type) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i = graph.input(t0.clone()).unwrap();
        let out = graph.get(i, s).unwrap();
        let t_result = worker.process_node(out).unwrap();
        assert_eq!(t_result, t1);
    }

    fn test_get_worker_fail(t0: Type, s: ArrayShape) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i = graph.input(t0.clone()).unwrap();
        let out = graph.get(i, s).unwrap();
        let e = worker.process_node(out);
        assert!(e.is_err());
    }

    #[test]
    fn test_get() {
        test_get_worker(
            array_type(vec![10, 20, 30], INT32),
            vec![],
            array_type(vec![10, 20, 30], INT32),
        );
        test_get_worker(
            array_type(vec![10, 20, 30], INT32),
            vec![7, 3],
            array_type(vec![30], INT32),
        );
        test_get_worker(
            array_type(vec![10, 20, 30], INT32),
            vec![7, 3, 2],
            scalar_type(INT32),
        );
        test_get_worker_fail(array_type(vec![10, 20, 30], INT32), vec![0, 0, 0, 0]);
        test_get_worker_fail(array_type(vec![10, 20, 30], INT32), vec![0, 5, 30]);
        test_get_worker_fail(scalar_type(INT32), vec![]);
        test_get_worker_fail(tuple_type(vec![]), vec![]);
    }

    fn test_get_slice_worker(t0: Type, s: Slice, t1: Type) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i = graph.input(t0.clone()).unwrap();
        let out = graph.get_slice(i, s).unwrap();
        let t_result = worker.process_node(out).unwrap();
        assert_eq!(t_result, t1);
    }

    fn test_get_slice_worker_fail(t0: Type, s: Slice) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i = graph.input(t0.clone()).unwrap();
        let out = i.get_slice(s).unwrap();
        assert!(worker.process_node(out).is_err());
    }

    #[test]
    fn test_get_slice() {
        test_get_slice_worker(
            array_type(vec![10, 20, 30], INT32),
            vec![],
            array_type(vec![10, 20, 30], INT32),
        );
        test_get_slice_worker(
            array_type(vec![10, 20, 30], INT32),
            vec![SliceElement::SingleIndex(3)],
            array_type(vec![20, 30], INT32),
        );
        test_get_slice_worker(
            array_type(vec![10, 20, 30], INT32),
            vec![SliceElement::SubArray(Some(7), None, Some(-3))],
            array_type(vec![3, 20, 30], INT32),
        );
        test_get_slice_worker(
            array_type(vec![10, 20, 30], INT32),
            vec![
                SliceElement::SingleIndex(3),
                SliceElement::SingleIndex(3),
                SliceElement::SingleIndex(3),
            ],
            scalar_type(INT32),
        );
        test_get_slice_worker_fail(tuple_type(vec![]), vec![]);
        test_get_slice_worker_fail(
            array_type(vec![10, 20, 30], INT32),
            vec![SliceElement::SingleIndex(123)],
        );
    }

    fn test_reshape_worker(t0: Type, new_type: Type) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i = graph.input(t0.clone()).unwrap();
        let out = graph.reshape(i, new_type.clone()).unwrap();
        let t_result = worker.process_node(out).unwrap();
        assert_eq!(t_result, new_type);
    }

    fn test_reshape_worker_fail(t0: Type, new_type: Type) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i = graph.input(t0.clone()).unwrap();
        let out_result = graph.reshape(i, new_type);
        if let Ok(out) = out_result {
            let e = worker.process_node(out);
            assert!(e.is_err());
        }
    }
    fn test_reshape_worker_unsuccessful(t0: Type, new_type: Type) {
        let context = create_unchecked_context().unwrap();
        let graph = context.create_graph().unwrap();
        let i = graph.input(t0.clone()).unwrap();
        let out_result = graph.reshape(i, new_type);
        assert!(out_result.is_err());
    }

    #[test]
    fn test_reshape() {
        test_reshape_worker(array_type(vec![4, 3, 5], BIT), array_type(vec![60], BIT));
        test_reshape_worker(
            array_type(vec![4, 3, 5], BIT),
            array_type(vec![1, 1, 1, 1, 60], BIT),
        );
        test_reshape_worker(array_type(vec![4, 3, 5], BIT), array_type(vec![15, 4], BIT));
        test_reshape_worker(
            array_type(vec![4, 3, 5], INT32),
            array_type(vec![2, 30], INT32),
        );
        test_reshape_worker_fail(
            array_type(vec![4, 3, 5], BIT),
            array_type(vec![1000000000, 10000000000, 1000000000], BIT),
        );
        assert!(!can_atomic_reshape(
            array_type(vec![4, 3, 5], BIT),
            array_type(vec![1000000000, 10000000000, 1000000000], BIT)
        ));
        assert!(!can_atomic_reshape(
            array_type(vec![1000000000, 10000000000, 1000000000], BIT),
            array_type(vec![4, 3, 5], BIT)
        ));
        test_reshape_worker_fail(tuple_type(vec![]), array_type(vec![], BIT));
        test_reshape_worker_fail(array_type(vec![4, 3, 5], BIT), array_type(vec![1, 63], BIT));
        test_reshape_worker(
            vector_type(3, array_type(vec![3, 4, 5], INT32)),
            tuple_type(vec![
                array_type(vec![3, 4, 5], INT32),
                array_type(vec![1, 30, 2], INT32),
                array_type(vec![60, 1], INT32),
            ]),
        );
        test_reshape_worker(
            vector_type(3, array_type(vec![3, 4, 5], INT32)),
            named_tuple_type(vec![
                ("field 1".to_string(), array_type(vec![3, 4, 5], INT32)),
                ("field 2".to_string(), array_type(vec![1, 30, 2], INT32)),
                ("field 3".to_string(), array_type(vec![60, 1], INT32)),
            ]),
        );
        test_reshape_worker(
            vector_type(0, array_type(vec![3, 4, 5], INT32)),
            tuple_type(vec![]),
        );
        test_reshape_worker_fail(array_type(vec![3, 4, 5], INT32), scalar_type(UINT32));
        test_reshape_worker_fail(scalar_type(UINT32), array_type(vec![3, 4, 5], INT32));
        test_reshape_worker(scalar_type(INT32), array_type(vec![1, 1, 1], INT32));
        test_reshape_worker(array_type(vec![1, 1, 1], INT32), scalar_type(INT32));
        test_reshape_worker(
            tuple_type(vec![
                array_type(vec![2], INT32),
                vector_type(
                    3,
                    tuple_type(vec![array_type(vec![2], INT32), array_type(vec![2], INT32)]),
                ),
                array_type(vec![2], INT32),
            ]),
            vector_type(8, array_type(vec![2], INT32)),
        );
        test_reshape_worker_fail(
            array_type(vec![3, 4, 5], INT32),
            vector_type(
                123456789,
                vector_type(123456789, vector_type(123456789, scalar_type(INT32))),
            ),
        );
        test_reshape_worker_unsuccessful(tuple_type(vec![]), array_type(vec![], BIT));
        test_reshape_worker_unsuccessful(
            tuple_type(vec![]),
            vector_type(u64::MAX, tuple_type(vec![])),
        );
    }

    fn test_nop_worker(t: Type) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i = graph.input(t.clone()).unwrap();
        let out = graph.nop(i).unwrap();
        let t_result = worker.process_node(out).unwrap();
        assert_eq!(t_result, t);
    }

    #[test]
    fn test_nop() {
        let t1 = array_type(vec![10, 10, 10], BIT);
        let t2 = tuple_type(vec![]);
        let t = named_tuple_type(vec![("field 1".to_owned(), t1), ("field 2".to_owned(), t2)]);
        test_nop_worker(t.clone());
    }

    fn test_random_worker(t: Type) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let out = graph.random(t.clone()).unwrap();
        let t_result = worker.process_node(out).unwrap();
        assert_eq!(t_result, t);
    }

    fn test_random_worker_fail(t: Type) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let out = graph.random(t.clone()).unwrap();
        let e = worker.process_node(out);
        assert!(e.is_err());
    }

    #[test]
    fn test_random() {
        let t0 = array_type(vec![31337], BIT);
        let t = tuple_type(vec![t0.clone(), t0.clone()]);
        test_random_worker(t);
        let t1 = array_type(vec![0, 0, 0], BIT);
        test_random_worker_fail(t1);
    }

    fn test_prf_worker(t0: Type, iv: u64, t1: Type) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let key = graph.input(t0).unwrap();
        let out = graph.prf(key, iv, t1.clone()).unwrap();
        let t_result = worker.process_node(out).unwrap();
        assert_eq!(t1, t_result);
    }

    fn test_prf_worker_fail(t0: Type, iv: u64, t1: Type) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let key = graph.input(t0).unwrap();
        let out = graph.prf(key, iv, t1.clone()).unwrap();
        let e = worker.process_node(out);
        assert!(e.is_err());
    }

    #[test]
    fn test_prf() {
        test_prf_worker(
            array_type(vec![128], BIT),
            31337,
            vector_type(123, tuple_type(vec![])),
        );
        test_prf_worker_fail(
            array_type(vec![127], BIT),
            31337,
            vector_type(123, tuple_type(vec![])),
        );
        test_prf_worker_fail(
            array_type(vec![128], INT32),
            31337,
            vector_type(123, tuple_type(vec![])),
        );
        test_prf_worker_fail(
            tuple_type(vec![]),
            31337,
            vector_type(123, tuple_type(vec![])),
        );
    }

    fn test_stack_worker(outer_shape: ArrayShape, inner_types: Vec<Type>, result_type: Type) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let inputs = inner_types
            .iter()
            .map(|x| graph.input(x.clone()).unwrap())
            .collect();
        let out = graph.stack(inputs, outer_shape).unwrap();
        let out_t = worker.process_node(out).unwrap();
        assert_eq!(out_t, result_type);
    }

    fn test_stack_worker_fail(outer_shape: ArrayShape, inner_types: Vec<Type>) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let inputs = inner_types
            .iter()
            .map(|x| graph.input(x.clone()).unwrap())
            .collect();
        let out = graph.stack(inputs, outer_shape).unwrap();
        let e = worker.process_node(out);
        assert!(e.is_err());
    }

    #[test]
    fn test_stack() {
        test_stack_worker(
            vec![2, 3],
            vec![
                scalar_type(BIT),
                scalar_type(BIT),
                scalar_type(BIT),
                scalar_type(BIT),
                scalar_type(BIT),
                scalar_type(BIT),
            ],
            array_type(vec![2, 3], BIT),
        );
        test_stack_worker(
            vec![2, 3],
            vec![
                scalar_type(BIT),
                scalar_type(BIT),
                array_type(vec![7, 5], BIT),
                scalar_type(BIT),
                scalar_type(BIT),
                scalar_type(BIT),
            ],
            array_type(vec![2, 3, 7, 5], BIT),
        );
        test_stack_worker_fail(
            vec![1000000000, 1000000000, 1000000000],
            vec![
                scalar_type(BIT),
                scalar_type(BIT),
                scalar_type(BIT),
                scalar_type(BIT),
                scalar_type(BIT),
                scalar_type(BIT),
            ],
        );
        test_stack_worker_fail(
            vec![2, 5, 8],
            vec![
                scalar_type(BIT),
                scalar_type(BIT),
                scalar_type(BIT),
                scalar_type(BIT),
                scalar_type(BIT),
                scalar_type(BIT),
            ],
        );
    }

    fn test_constant_worker(t: Type, v: Value) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let out = graph.constant(t.clone(), v).unwrap();
        let out_t = worker.process_node(out).unwrap();
        assert_eq!(out_t, t);
    }

    fn test_constant_worker_fail(t: Type, v: Value) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let out = graph.constant(t.clone(), v).unwrap();
        let e = worker.process_node(out);
        assert!(e.is_err());
    }

    #[test]
    fn test_constant() {
        let v = Value::from_vector(vec![
            Value::from_bytes(vec![0]),
            Value::from_bytes(vec![0, 0, 0, 0]),
        ]);
        let t = tuple_type(vec![scalar_type(BIT), scalar_type(INT32)]);
        let t1 = tuple_type(vec![]);
        test_constant_worker(t, v.clone());
        test_constant_worker_fail(t1, v);
    }

    fn test_a2b_worker(t0: Type, t1: Type) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i = graph.input(t0).unwrap();
        let o = graph.a2b(i).unwrap();
        let o_t = worker.process_node(o).unwrap();
        assert_eq!(o_t, t1);
    }

    fn test_a2b_worker_fail(t0: Type) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i = graph.input(t0).unwrap();
        let o = graph.a2b(i).unwrap();
        let e = worker.process_node(o);
        assert!(e.is_err());
    }

    #[test]
    fn test_a2b() {
        test_a2b_worker(scalar_type(INT32), array_type(vec![32], BIT));
        test_a2b_worker(
            array_type(vec![10, 20], INT32),
            array_type(vec![10, 20, 32], BIT),
        );
        test_a2b_worker_fail(array_type(vec![10, 20], BIT));
        test_a2b_worker_fail(tuple_type(vec![]));
    }

    fn test_b2a_worker(t0: Type, st: ScalarType, t1: Type) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i = graph.input(t0).unwrap();
        let o = graph.b2a(i, st).unwrap();
        let o_t = worker.process_node(o).unwrap();
        assert_eq!(o_t, t1);
    }

    fn test_b2a_worker_fail(t0: Type, st: ScalarType) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i = graph.input(t0).unwrap();
        let o = graph.b2a(i, st).unwrap();
        let e = worker.process_node(o);
        assert!(e.is_err());
    }

    use crate::constants::type_size_limit_constants;
    #[test]
    fn test_b2a() {
        test_b2a_worker(
            array_type(vec![10, 32], BIT),
            INT32,
            array_type(vec![10], INT32),
        );
        test_b2a_worker(array_type(vec![32], BIT), INT32, scalar_type(INT32));
        test_b2a_worker_fail(tuple_type(vec![]), INT32);
        test_b2a_worker_fail(array_type(vec![10, 20, 1], BIT), BIT);
        test_b2a_worker_fail(array_type(vec![10, 20, 1], INT32), INT32);
        test_b2a_worker_fail(array_type(vec![10, 40], BIT), INT32);
        if type_size_limit_constants::NON_STANDARD_SCALAR_LEN_SUPPORT {
            let t = create_scalar_type(false, Some(126));
            test_b2a_worker(array_type(vec![7], BIT), t.clone(), scalar_type(t.clone()));
            test_b2a_worker(
                array_type(vec![123, 45, 7], BIT),
                t.clone(),
                array_type(vec![123, 45], t.clone()),
            );
        }
    }

    fn test_create_tuple_worker(elements: Vec<Type>, expected_result: Type) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let nodes: Vec<Node> = elements
            .iter()
            .map(|node| graph.input(node.clone()).unwrap())
            .collect();
        let output = graph.create_tuple(nodes).unwrap();
        let t = worker.process_node(output).unwrap();
        assert_eq!(expected_result, t);
    }

    #[test]
    fn test_create_tuple() {
        test_create_tuple_worker(vec![], tuple_type(vec![]));
        test_create_tuple_worker(vec![scalar_type(BIT)], tuple_type(vec![scalar_type(BIT)]));
        test_create_tuple_worker(
            vec![
                scalar_type(BIT),
                scalar_type(INT32),
                tuple_type(vec![]),
                array_type(vec![10, 20], INT32),
            ],
            tuple_type(vec![
                scalar_type(BIT),
                scalar_type(INT32),
                tuple_type(vec![]),
                array_type(vec![10, 20], INT32),
            ]),
        );
    }

    fn test_create_named_tuple_worker(
        elements: Vec<Type>,
        names: Vec<String>,
        expected_result: Type,
    ) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let nodes: Vec<Node> = elements
            .iter()
            .map(|node| graph.input(node.clone()).unwrap())
            .collect();
        let fused = names
            .iter()
            .zip(nodes.iter())
            .map(|(x, y)| (x.clone(), y.clone()))
            .collect();
        let output = graph.create_named_tuple(fused).unwrap();
        let t = worker.process_node(output).unwrap();
        assert_eq!(expected_result, t);
    }

    fn test_create_named_tuple_worker_fail(elements: Vec<Type>, names: Vec<String>) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let nodes: Vec<Node> = elements
            .iter()
            .map(|node| graph.input(node.clone()).unwrap())
            .collect();
        let fused = names
            .iter()
            .zip(nodes.iter())
            .map(|(x, y)| (x.clone(), y.clone()))
            .collect();
        let output = graph.create_named_tuple(fused).unwrap();
        let e = worker.process_node(output);
        assert!(e.is_err());
    }

    #[test]
    fn test_create_named_tuple() {
        test_create_named_tuple_worker(vec![], vec![], named_tuple_type(vec![]));
        test_create_named_tuple_worker(
            vec![scalar_type(BIT)],
            vec!["field 1".to_owned()],
            named_tuple_type(vec![("field 1".to_owned(), scalar_type(BIT))]),
        );
        test_create_named_tuple_worker_fail(
            vec![scalar_type(BIT), tuple_type(vec![])],
            vec!["field 1".to_owned(), "field 1".to_owned()],
        );
    }

    fn test_create_vector_worker(element_type: Type, elements: Vec<Type>) -> Result<Type> {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let nodes: Vec<Node> = elements
            .iter()
            .map(|node| graph.input(node.clone()).unwrap())
            .collect();
        let output = graph.create_vector(element_type, nodes).unwrap();
        return worker.process_node(output);
    }

    #[test]
    fn test_create_vector() {
        assert_eq!(
            test_create_vector_worker(scalar_type(BIT), vec![]).unwrap(),
            vector_type(0, scalar_type(BIT))
        );
        assert_eq!(
            test_create_vector_worker(scalar_type(BIT), vec![scalar_type(BIT), scalar_type(BIT)])
                .unwrap(),
            vector_type(2, scalar_type(BIT))
        );
        assert!(test_create_vector_worker(
            scalar_type(BIT),
            vec![scalar_type(BIT), scalar_type(UINT32)]
        )
        .is_err());
        assert!(test_create_vector_worker(scalar_type(BIT), vec![scalar_type(UINT32)]).is_err());
    }

    fn test_tuple_get_worker(t0: Type, ind: u64, t1: Type) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i = graph.input(t0.clone()).unwrap();
        let o = graph.tuple_get(i.clone(), ind).unwrap();
        let out_t = worker.process_node(o).unwrap();
        assert_eq!(out_t, t1);
    }

    fn test_tuple_get_worker_fail(t0: Type, ind: u64) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i = graph.input(t0.clone()).unwrap();
        let o = graph.tuple_get(i.clone(), ind).unwrap();
        let e = worker.process_node(o);
        assert!(e.is_err());
    }

    #[test]
    fn test_tuple_get() {
        test_tuple_get_worker(
            tuple_type(vec![scalar_type(BIT), scalar_type(INT32)]),
            0,
            scalar_type(BIT),
        );
        test_tuple_get_worker(
            tuple_type(vec![scalar_type(BIT), scalar_type(INT32)]),
            1,
            scalar_type(INT32),
        );
        test_tuple_get_worker(
            named_tuple_type(vec![
                ("field 1".to_owned(), scalar_type(BIT)),
                ("field 2".to_owned(), scalar_type(INT32)),
            ]),
            0,
            scalar_type(BIT),
        );
        test_tuple_get_worker(
            named_tuple_type(vec![
                ("field 1".to_owned(), scalar_type(BIT)),
                ("field 2".to_owned(), scalar_type(INT32)),
            ]),
            1,
            scalar_type(INT32),
        );
        test_tuple_get_worker_fail(tuple_type(vec![]), 0);
        test_tuple_get_worker_fail(named_tuple_type(vec![]), 0);
        test_tuple_get_worker_fail(array_type(vec![10, 10], BIT), 0);
    }

    fn test_named_tuple_get_worker(t0: Type, field_name: String, t1: Type) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i = graph.input(t0.clone()).unwrap();
        let o = graph.named_tuple_get(i.clone(), field_name).unwrap();
        let out_t = worker.process_node(o).unwrap();
        assert_eq!(out_t, t1);
    }

    fn test_named_tuple_get_worker_fail(t0: Type, field_name: String) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i = graph.input(t0.clone()).unwrap();
        let o = graph.named_tuple_get(i.clone(), field_name).unwrap();
        let e = worker.process_node(o);
        assert!(e.is_err());
    }

    fn test_vector_get_worker(t0: Type, t1: Type, expected_out_t: Type) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i0 = graph.input(t0.clone()).unwrap();
        let i1 = graph.input(t1.clone()).unwrap();
        let o = graph.vector_get(i0.clone(), i1.clone()).unwrap();
        let out_t = worker.process_node(o).unwrap();
        assert_eq!(out_t, expected_out_t);
    }

    fn test_vector_get_worker_fail(t0: Type, t1: Type) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i0 = graph.input(t0.clone()).unwrap();
        let i1 = graph.input(t1.clone()).unwrap();
        let o = graph.vector_get(i0.clone(), i1.clone()).unwrap();
        let e = worker.process_node(o);
        assert!(e.is_err());
    }

    #[test]
    fn test_vector_get() {
        test_vector_get_worker(
            vector_type(10, scalar_type(BIT)),
            scalar_type(UINT64),
            scalar_type(BIT),
        );
        test_vector_get_worker(
            vector_type(10, scalar_type(INT32)),
            scalar_type(UINT64),
            scalar_type(INT32),
        );
        test_vector_get_worker_fail(vector_type(10, scalar_type(INT32)), scalar_type(BIT));
        test_vector_get_worker_fail(
            vector_type(10, scalar_type(INT32)),
            vector_type(1, scalar_type(UINT64)),
        );
        test_vector_get_worker_fail(
            tuple_type(vec![scalar_type(BIT), scalar_type(BIT)]),
            scalar_type(UINT64),
        );
        test_vector_get_worker_fail(array_type(vec![10, 10], BIT), scalar_type(UINT64));
        test_vector_get_worker_fail(scalar_type(BIT), scalar_type(UINT64));
    }

    #[test]
    fn test_named_tuple_get() {
        test_named_tuple_get_worker(
            named_tuple_type(vec![
                ("field 1".to_owned(), scalar_type(BIT)),
                ("field 2".to_owned(), scalar_type(INT32)),
            ]),
            "field 1".to_owned(),
            scalar_type(BIT),
        );
        test_named_tuple_get_worker(
            named_tuple_type(vec![
                ("field 1".to_owned(), scalar_type(BIT)),
                ("field 2".to_owned(), scalar_type(INT32)),
            ]),
            "field 2".to_owned(),
            scalar_type(INT32),
        );
        test_named_tuple_get_worker_fail(
            named_tuple_type(vec![
                ("field 1".to_owned(), scalar_type(BIT)),
                ("field 2".to_owned(), scalar_type(INT32)),
            ]),
            "field 0".to_owned(),
        );
        test_named_tuple_get_worker_fail(scalar_type(BIT), "field 0".to_owned());
    }

    fn test_zip_worker(t0: Vec<Type>, t1: Type) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let nodes = t0.iter().map(|t| graph.input(t.clone()).unwrap()).collect();
        let o = graph.zip(nodes).unwrap();
        let out_t = worker.process_node(o).unwrap();
        assert_eq!(out_t, t1);
    }

    fn test_zip_worker_fail(t0: Vec<Type>) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let nodes = t0.iter().map(|t| graph.input(t.clone()).unwrap()).collect();
        let o = graph.zip(nodes).unwrap();
        let e = worker.process_node(o);
        assert!(e.is_err());
    }

    #[test]
    fn test_zip() {
        test_zip_worker(
            vec![
                vector_type(0, scalar_type(BIT)),
                vector_type(0, scalar_type(INT32)),
            ],
            vector_type(0, tuple_type(vec![scalar_type(BIT), scalar_type(INT32)])),
        );
        test_zip_worker(
            vec![
                vector_type(10, scalar_type(BIT)),
                vector_type(10, scalar_type(INT32)),
            ],
            vector_type(10, tuple_type(vec![scalar_type(BIT), scalar_type(INT32)])),
        );
        test_zip_worker(
            vec![
                vector_type(10, scalar_type(BIT)),
                vector_type(10, scalar_type(INT32)),
                vector_type(10, array_type(vec![10, 10], INT32)),
            ],
            vector_type(
                10,
                tuple_type(vec![
                    scalar_type(BIT),
                    scalar_type(INT32),
                    array_type(vec![10, 10], INT32),
                ]),
            ),
        );
        test_zip_worker_fail(vec![
            vector_type(10, scalar_type(BIT)),
            vector_type(10, scalar_type(INT32)),
            vector_type(8, array_type(vec![10, 10], INT32)),
        ]);
        test_zip_worker_fail(vec![
            vector_type(10, scalar_type(BIT)),
            scalar_type(BIT),
            vector_type(10, array_type(vec![10, 10], INT32)),
        ]);
        test_zip_worker_fail(vec![]);
        test_zip_worker_fail(vec![vector_type(10, scalar_type(BIT))]);
    }

    fn test_repeat_worker(t0: Type, n: u64, t1: Type) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i = graph.input(t0).unwrap();
        let o = graph.repeat(i.clone(), n).unwrap();
        let o_t = worker.process_node(o).unwrap();
        assert_eq!(o_t, t1);
    }

    #[test]
    fn test_repeat() {
        test_repeat_worker(scalar_type(BIT), 0, vector_type(0, scalar_type(BIT)));
        test_repeat_worker(scalar_type(BIT), 10, vector_type(10, scalar_type(BIT)));
    }

    fn create_dummy_graph(input_types: Vec<Type>, output_type: Type, context: Context) -> Graph {
        let graph = context.create_graph().unwrap();
        let _input_nodes: Vec<Node> = input_types
            .iter()
            .map(|t| graph.input(t.clone()).unwrap())
            .collect();
        let output_node = graph
            .constant(
                output_type.clone(),
                Value::zero_of_type(output_type.clone()),
            )
            .unwrap();
        graph.set_output_node(output_node.clone()).unwrap();
        graph.finalize().unwrap();
        graph
    }

    fn test_call_worker(input_types: Vec<Type>, output_type: Type) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let subroutine_graph =
            create_dummy_graph(input_types.clone(), output_type.clone(), context.clone());
        let graph = context.create_graph().unwrap();
        let input_nodes: Vec<Node> = input_types
            .iter()
            .map(|t| graph.input(t.clone()).unwrap())
            .collect();
        let output_node = graph
            .call(subroutine_graph.clone(), input_nodes.clone())
            .unwrap();
        let out_t = worker.process_node(output_node).unwrap();
        assert_eq!(out_t, output_type);
    }

    fn test_call_worker_fail(
        input_types_subroutine: Vec<Type>,
        output_type: Type,
        input_types_actual: Vec<Type>,
    ) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let subroutine_graph = create_dummy_graph(
            input_types_subroutine.clone(),
            output_type.clone(),
            context.clone(),
        );
        let graph = context.create_graph().unwrap();
        let input_nodes: Vec<Node> = input_types_actual
            .iter()
            .map(|t| graph.input(t.clone()).unwrap())
            .collect();
        let output_node = graph
            .call(subroutine_graph.clone(), input_nodes.clone())
            .unwrap();
        let e = worker.process_node(output_node);
        assert!(e.is_err());
    }

    #[test]
    fn test_call() {
        test_call_worker(vec![], scalar_type(BIT));
        test_call_worker(vec![scalar_type(BIT)], scalar_type(BIT));
        test_call_worker(vec![scalar_type(BIT), scalar_type(BIT)], scalar_type(BIT));
        test_call_worker(vec![scalar_type(BIT), scalar_type(INT32)], scalar_type(BIT));
        test_call_worker_fail(
            vec![scalar_type(BIT), scalar_type(INT32)],
            scalar_type(BIT),
            vec![scalar_type(BIT)],
        );
        test_call_worker_fail(
            vec![scalar_type(BIT)],
            scalar_type(BIT),
            vec![scalar_type(BIT), scalar_type(INT32)],
        );
        test_call_worker_fail(
            vec![scalar_type(BIT), scalar_type(BIT)],
            scalar_type(BIT),
            vec![scalar_type(BIT), scalar_type(INT32)],
        );
    }

    fn test_iterate_worker(
        subroutine_input_types: Vec<Type>,
        subroutine_output_type: Type,
        input_types: Vec<Type>,
        output_type: Type,
    ) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let subroutine_graph = create_dummy_graph(
            subroutine_input_types,
            subroutine_output_type,
            context.clone(),
        );
        let graph = context.create_graph().unwrap();
        let input_nodes: Vec<Node> = input_types
            .iter()
            .map(|t| graph.input(t.clone()).unwrap())
            .collect();
        assert_eq!(input_nodes.len(), 2);
        let output_node = graph
            .iterate(
                subroutine_graph,
                input_nodes[0].clone(),
                input_nodes[1].clone(),
            )
            .unwrap();
        let output_obtained_type = worker.process_node(output_node.clone()).unwrap();
        assert_eq!(output_obtained_type, output_type);
    }

    fn test_iterate_worker_fail(
        subroutine_input_types: Vec<Type>,
        subroutine_output_type: Type,
        input_types: Vec<Type>,
    ) {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let subroutine_graph = create_dummy_graph(
            subroutine_input_types,
            subroutine_output_type,
            context.clone(),
        );
        let graph = context.create_graph().unwrap();
        let input_nodes: Vec<Node> = input_types
            .iter()
            .map(|t| graph.input(t.clone()).unwrap())
            .collect();
        assert_eq!(input_nodes.len(), 2);
        let output_node = graph
            .iterate(
                subroutine_graph,
                input_nodes[0].clone(),
                input_nodes[1].clone(),
            )
            .unwrap();
        let e = worker.process_node(output_node.clone());
        assert!(e.is_err());
    }

    #[test]
    fn test_iterate() {
        test_iterate_worker(
            vec![scalar_type(BIT), scalar_type(INT8)],
            tuple_type(vec![scalar_type(BIT), scalar_type(INT32)]),
            vec![scalar_type(BIT), vector_type(4057, scalar_type(INT8))],
            tuple_type(vec![
                scalar_type(BIT),
                vector_type(4057, scalar_type(INT32)),
            ]),
        );
        test_iterate_worker_fail(
            vec![scalar_type(BIT)],
            tuple_type(vec![scalar_type(BIT), scalar_type(INT32)]),
            vec![scalar_type(BIT), vector_type(4057, scalar_type(INT8))],
        );
        test_iterate_worker_fail(
            vec![scalar_type(BIT), scalar_type(INT8)],
            tuple_type(vec![scalar_type(INT32), scalar_type(INT32)]),
            vec![scalar_type(BIT), vector_type(4057, scalar_type(INT8))],
        );
        test_iterate_worker_fail(
            vec![scalar_type(BIT), scalar_type(INT8)],
            scalar_type(BIT),
            vec![scalar_type(BIT), vector_type(4057, scalar_type(INT8))],
        );
        test_iterate_worker_fail(
            vec![scalar_type(BIT), scalar_type(INT8)],
            tuple_type(vec![scalar_type(BIT), scalar_type(INT32), scalar_type(BIT)]),
            vec![scalar_type(BIT), vector_type(4057, scalar_type(INT8))],
        );
        test_iterate_worker_fail(
            vec![scalar_type(BIT), scalar_type(INT8)],
            tuple_type(vec![scalar_type(BIT), scalar_type(INT32)]),
            vec![scalar_type(BIT), scalar_type(BIT)],
        );
        test_iterate_worker_fail(
            vec![scalar_type(BIT), scalar_type(INT8)],
            tuple_type(vec![scalar_type(BIT), scalar_type(INT32)]),
            vec![scalar_type(BIT), vector_type(4057, scalar_type(INT32))],
        );
        test_iterate_worker_fail(
            vec![scalar_type(BIT), scalar_type(INT8)],
            tuple_type(vec![scalar_type(BIT), scalar_type(INT32)]),
            vec![scalar_type(INT32), vector_type(4057, scalar_type(INT8))],
        );
    }

    fn test_array_to_vector_worker(t0: Type, t1: Type) {
        let context = create_unchecked_context().unwrap();
        let graph = context.create_graph().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let i = graph.input(t0).unwrap();
        let o = graph.array_to_vector(i).unwrap();
        let t = worker.process_node(o).unwrap();
        assert_eq!(t, t1);
    }

    fn test_array_to_vector_worker_fail(t0: Type) {
        let context = create_unchecked_context().unwrap();
        let graph = context.create_graph().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let i = graph.input(t0).unwrap();
        let o = graph.array_to_vector(i).unwrap();
        let e = worker.process_node(o);
        assert!(e.is_err());
    }

    #[test]
    fn test_array_to_vector() {
        test_array_to_vector_worker(
            array_type(vec![10, 3, 8], BIT),
            vector_type(10, array_type(vec![3, 8], BIT)),
        );
        test_array_to_vector_worker(array_type(vec![10], BIT), vector_type(10, scalar_type(BIT)));
        test_array_to_vector_worker_fail(scalar_type(BIT));
        test_array_to_vector_worker_fail(tuple_type(vec![]));
    }

    fn test_vector_to_array_worker(t0: Type, t1: Type) {
        let context = create_unchecked_context().unwrap();
        let graph = context.create_graph().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let i = graph.input(t0).unwrap();
        let o = graph.vector_to_array(i).unwrap();
        let t = worker.process_node(o).unwrap();
        assert_eq!(t, t1);
    }

    fn test_vector_to_array_worker_fail(t0: Type) {
        let context = create_unchecked_context().unwrap();
        let graph = context.create_graph().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let i = graph.input(t0).unwrap();
        let o = graph.vector_to_array(i).unwrap();
        let e = worker.process_node(o);
        assert!(e.is_err());
    }

    #[test]
    fn test_vector_to_array() {
        test_vector_to_array_worker(
            vector_type(10, array_type(vec![3, 8], BIT)),
            array_type(vec![10, 3, 8], BIT),
        );
        test_vector_to_array_worker(vector_type(10, scalar_type(BIT)), array_type(vec![10], BIT));
        test_vector_to_array_worker_fail(vector_type(0, array_type(vec![3, 8], BIT)));
        test_vector_to_array_worker_fail(tuple_type(vec![]));
        test_vector_to_array_worker_fail(vector_type(10, tuple_type(vec![])));
    }

    fn gather_helper(
        input_t: Type,
        indices_t: Type,
        axis: u64,
        expected: Option<Type>,
    ) -> Result<()> {
        let context = create_unchecked_context()?;
        let graph = context.create_graph()?;
        let mut worker = create_type_inference_worker(context.clone());
        let inp = graph.input(input_t)?;
        let ind = graph.input(indices_t)?;
        let o = graph.gather(inp, ind, axis)?;
        let t = worker.process_node(o);
        if let Some(expected_t) = expected {
            assert_eq!(t?, expected_t);
        } else {
            assert!(t.is_err());
        }
        Ok(())
    }

    #[test]
    fn test_gather() {
        || -> Result<()> {
            gather_helper(
                array_type(vec![2, 3, 4], BIT),
                array_type(vec![2], UINT64),
                1,
                Some(array_type(vec![2, 2, 4], BIT)),
            )?;

            gather_helper(
                array_type(vec![4], BIT),
                array_type(vec![3], UINT64),
                0,
                Some(array_type(vec![3], BIT)),
            )?;

            gather_helper(
                array_type(vec![2, 3, 7, 5], BIT),
                array_type(vec![2, 3], UINT64),
                2,
                Some(array_type(vec![2, 3, 2, 3, 5], BIT)),
            )?;

            gather_helper(scalar_type(BIT), array_type(vec![2], UINT64), 1, None)?;
            gather_helper(array_type(vec![2, 3, 4], BIT), scalar_type(UINT64), 1, None)?;
            gather_helper(
                array_type(vec![2, 3, 4], BIT),
                array_type(vec![2], UINT32),
                1,
                None,
            )?;
            gather_helper(
                array_type(vec![2, 3, 4], BIT),
                array_type(vec![2], UINT64),
                3,
                None,
            )?;
            gather_helper(
                array_type(vec![2, 3, 4], BIT),
                array_type(vec![2, 2], UINT64),
                1,
                None,
            )?;

            Ok(())
        }()
        .unwrap();
    }

    fn test_cuckoo_hash_worker(t0: Type, t1: Type, expected: Type) -> Result<()> {
        let context = create_unchecked_context()?;
        let graph = context.create_graph()?;
        let mut worker = create_type_inference_worker(context.clone());
        let i = graph.input(t0)?;
        let h = graph.input(t1)?;
        let o = graph.cuckoo_hash(i, h)?;
        let t = worker.process_node(o)?;
        assert_eq!(t, expected);
        Ok(())
    }

    fn test_cuckoo_hash_fail(t0: Type, t1: Type) -> Result<()> {
        let context = create_unchecked_context()?;
        let graph = context.create_graph()?;
        let mut worker = create_type_inference_worker(context.clone());
        let i = graph.input(t0)?;
        let h = graph.input(t1)?;
        let o = graph.cuckoo_hash(i, h)?;
        let t = worker.process_node(o);
        assert!(t.is_err());
        Ok(())
    }

    #[test]
    fn test_cuckoo_hash() {
        || -> Result<()> {
            test_cuckoo_hash_worker(
                array_type(vec![5, 6], BIT),
                array_type(vec![3, 4, 6], BIT),
                array_type(vec![16], UINT64),
            )?;
            test_cuckoo_hash_worker(
                array_type(vec![4, 6], BIT),
                array_type(vec![3, 3, 6], BIT),
                array_type(vec![8], UINT64),
            )?;
            test_cuckoo_hash_worker(
                array_type(vec![11, 4, 6], BIT),
                array_type(vec![4, 5, 6], BIT),
                array_type(vec![11, 32], UINT64),
            )?;

            test_cuckoo_hash_fail(scalar_type(BIT), array_type(vec![3, 3, 6], BIT))?;
            test_cuckoo_hash_fail(array_type(vec![4, 6], INT8), array_type(vec![3, 4, 6], BIT))?;
            test_cuckoo_hash_fail(array_type(vec![6], BIT), array_type(vec![3, 4, 8], BIT))?;
            test_cuckoo_hash_fail(
                array_type(vec![4, 6], BIT),
                vector_type(2, array_type(vec![3, 4, 6], BIT)),
            )?;
            test_cuckoo_hash_fail(
                array_type(vec![4, 6], BIT),
                array_type(vec![3, 4, 6], UINT64),
            )?;
            test_cuckoo_hash_fail(array_type(vec![4, 6], BIT), array_type(vec![3, 6], BIT))?;
            test_cuckoo_hash_fail(array_type(vec![4, 6], BIT), array_type(vec![2, 4, 6], BIT))?;
            test_cuckoo_hash_fail(array_type(vec![4, 6], BIT), array_type(vec![3, 4, 7], BIT))?;
            test_cuckoo_hash_fail(array_type(vec![4, 6], BIT), array_type(vec![3, 64, 6], BIT))?;

            Ok(())
        }()
        .unwrap();
    }

    fn test_random_permutation_worker(n: u64) -> Result<Type> {
        let context = create_unchecked_context()?;
        let graph = context.create_graph()?;
        let mut worker = create_type_inference_worker(context.clone());
        let o = graph.random_permutation(n)?;
        worker.process_node(o)
    }

    #[test]
    fn test_random_permutation() {
        || -> Result<()> {
            assert_eq!(
                test_random_permutation_worker(1)?,
                array_type(vec![1], UINT64)
            );
            assert_eq!(
                test_random_permutation_worker(100)?,
                array_type(vec![100], UINT64)
            );

            assert!(test_random_permutation_worker(0).is_err());

            Ok(())
        }()
        .unwrap();
    }

    fn test_inverse_permutation_worker(t0: Type, expected: Type) -> Result<()> {
        let context = create_unchecked_context()?;
        let graph = context.create_graph()?;
        let mut worker = create_type_inference_worker(context.clone());
        let i = graph.input(t0)?;
        let o = graph.inverse_permutation(i)?;
        let t = worker.process_node(o)?;
        assert_eq!(t, expected);
        Ok(())
    }

    fn test_inverse_permutation_fail(t0: Type) -> Result<()> {
        let context = create_unchecked_context()?;
        let graph = context.create_graph()?;
        let mut worker = create_type_inference_worker(context.clone());
        let i = graph.input(t0)?;
        let o = graph.inverse_permutation(i)?;
        let t = worker.process_node(o);
        assert!(t.is_err());
        Ok(())
    }

    #[test]
    fn test_inverse_permutation() {
        || -> Result<()> {
            let t = array_type(vec![10], UINT64);
            test_inverse_permutation_worker(t.clone(), t)?;

            test_inverse_permutation_fail(scalar_type(UINT64))?;
            test_inverse_permutation_fail(array_type(vec![10, 5], UINT64))?;
            test_inverse_permutation_fail(array_type(vec![10], UINT32))?;

            Ok(())
        }()
        .unwrap();
    }

    fn test_cuckoo_to_permutation_worker(t0: Type, expected: Type) -> Result<()> {
        let context = create_unchecked_context()?;
        let graph = context.create_graph()?;
        let mut worker = create_type_inference_worker(context.clone());
        let i = graph.input(t0)?;
        let o = graph.cuckoo_to_permutation(i)?;
        let t = worker.process_node(o)?;
        assert_eq!(t, expected);
        Ok(())
    }

    fn test_cuckoo_to_permutation_fail(t0: Type) -> Result<()> {
        let context = create_unchecked_context()?;
        let graph = context.create_graph()?;
        let mut worker = create_type_inference_worker(context.clone());
        let i = graph.input(t0)?;
        let o = graph.cuckoo_to_permutation(i)?;
        let t = worker.process_node(o);
        assert!(t.is_err());
        Ok(())
    }

    #[test]
    fn test_cuckoo_to_permutation() {
        || -> Result<()> {
            let t = array_type(vec![2, 3, 10], UINT64);
            test_cuckoo_to_permutation_worker(t.clone(), t)?;
            let t = array_type(vec![10], UINT64);
            test_cuckoo_to_permutation_worker(t.clone(), t)?;

            test_cuckoo_to_permutation_fail(scalar_type(UINT64))?;
            test_cuckoo_to_permutation_fail(array_type(vec![10], UINT32))?;

            Ok(())
        }()
        .unwrap();
    }

    fn test_decompose_switching_map_worker(t: Type, n: u64) -> Result<()> {
        let context = create_unchecked_context()?;
        let graph = context.create_graph()?;
        let mut worker = create_type_inference_worker(context.clone());
        let i = graph.input(t.clone())?;
        let o = i.decompose_switching_map(n)?;
        let res_t = worker.process_node(o)?;
        let shape = t.get_shape();
        let duplication_map_t = tuple_type(vec![
            array_type(shape.clone(), UINT64),
            array_type(shape, BIT),
        ]);
        assert_eq!(res_t, tuple_type(vec![t.clone(), duplication_map_t, t]));
        Ok(())
    }

    fn test_decompose_switching_map_fail(t: Type, n: u64) -> Result<()> {
        let context = create_unchecked_context()?;
        let graph = context.create_graph()?;
        let mut worker = create_type_inference_worker(context.clone());
        let i = graph.input(t)?;
        let o = i.decompose_switching_map(n)?;
        let res_t = worker.process_node(o);
        assert!(res_t.is_err());
        Ok(())
    }

    #[test]
    fn test_decompose_switching_map() {
        || -> Result<()> {
            let t = array_type(vec![2, 3, 10], UINT64);
            test_decompose_switching_map_worker(t, 10)?;
            let t = array_type(vec![10], UINT64);
            test_decompose_switching_map_worker(t, 11)?;

            test_decompose_switching_map_fail(scalar_type(UINT64), 10)?;
            test_decompose_switching_map_fail(array_type(vec![10], UINT32), 10)?;
            test_decompose_switching_map_fail(array_type(vec![10], UINT64), 9)?;

            Ok(())
        }()
        .unwrap();
    }

    fn test_segment_cumsum_worker(shape: ArrayShape, st: ScalarType) -> Result<()> {
        let context = create_unchecked_context()?;
        let graph = context.create_graph()?;
        let mut worker = create_type_inference_worker(context.clone());
        let t = array_type(shape.clone(), st.clone());
        let i = graph.input(t.clone())?;
        let b = graph.input(array_type(vec![shape[0]], BIT))?;
        let v = if shape.len() > 1 {
            graph.input(array_type(shape[1..].to_vec(), st.clone()))?
        } else {
            graph.input(scalar_type(st.clone()))?
        };
        let o = i.segment_cumsum(b, v)?;
        let res_t = worker.process_node(o)?;
        let mut expected_shape = shape.clone();
        expected_shape[0] = shape[0] + 1;
        let expected_t = array_type(expected_shape, st);
        assert_eq!(res_t, expected_t);
        Ok(())
    }

    fn test_segment_cumsum_fail(input_t: Type, binary_t: Type, first_row_t: Type) -> Result<()> {
        let context = create_unchecked_context()?;
        let graph = context.create_graph()?;
        let mut worker = create_type_inference_worker(context.clone());
        let i = graph.input(input_t)?;
        let b = graph.input(binary_t)?;
        let v = graph.input(first_row_t)?;
        let o = i.segment_cumsum(b, v)?;
        let res_t = worker.process_node(o);
        assert!(res_t.is_err());
        Ok(())
    }

    #[test]
    fn test_segment_cumsum() {
        || -> Result<()> {
            test_segment_cumsum_worker(vec![5], BIT)?;
            test_segment_cumsum_worker(vec![5], UINT64)?;
            test_segment_cumsum_worker(vec![5, 16], BIT)?;
            test_segment_cumsum_worker(vec![5, 16], UINT64)?;

            test_segment_cumsum_fail(scalar_type(BIT), scalar_type(BIT), scalar_type(BIT))?;
            test_segment_cumsum_fail(array_type(vec![4], BIT), scalar_type(BIT), scalar_type(BIT))?;
            test_segment_cumsum_fail(
                array_type(vec![4], BIT),
                array_type(vec![3], BIT),
                scalar_type(BIT),
            )?;
            test_segment_cumsum_fail(
                array_type(vec![4], INT32),
                array_type(vec![4], INT32),
                scalar_type(INT32),
            )?;
            test_segment_cumsum_fail(
                array_type(vec![4], INT32),
                array_type(vec![4], BIT),
                scalar_type(BIT),
            )?;
            test_segment_cumsum_fail(
                array_type(vec![4, 4], INT32),
                array_type(vec![4], BIT),
                scalar_type(BIT),
            )?;

            Ok(())
        }()
        .unwrap();
    }

    struct ColumnDescription {
        header: String,
        row_shape: Vec<u64>,
        st: ScalarType,
    }

    fn create_column(header: &str, row_shape: Vec<u64>, st: ScalarType) -> ColumnDescription {
        ColumnDescription {
            header: header.to_owned(),
            row_shape,
            st,
        }
    }

    fn join_worker(
        columns0: Vec<ColumnDescription>,
        num0: u64,
        columns1: Vec<ColumnDescription>,
        num1: u64,
        op: Operation,
        expected_columns: Vec<ColumnDescription>,
    ) -> Result<()> {
        let context = create_unchecked_context()?;
        let graph = context.create_graph()?;
        let mut worker = create_type_inference_worker(context.clone());
        let create_named_tuple_type = |columns: Vec<ColumnDescription>, num: u64| -> Type {
            let mut named_tuples = vec![];
            for column_desc in columns {
                let mut column_shape = column_desc.row_shape;
                if column_shape == [1] {
                    column_shape[0] = num;
                } else {
                    column_shape.insert(0, num);
                }
                named_tuples.push((column_desc.header, array_type(column_shape, column_desc.st)));
            }
            named_tuple_type(named_tuples)
        };
        let i0 = graph.input(create_named_tuple_type(columns0, num0))?;
        let i1 = graph.input(create_named_tuple_type(columns1, num1))?;
        let o = graph.add_node(vec![i0, i1], vec![], op.clone())?;
        let res_t = worker.process_node(o)?;
        let num_expected = if let Operation::Join(join_t, _) = op {
            match join_t {
                JoinType::Inner | JoinType::Left => num0,
                JoinType::Union | JoinType::Full => num0 + num1,
            }
        } else {
            panic!("Shouldn't be here");
        };
        let expected_t = create_named_tuple_type(expected_columns, num_expected);
        assert_eq!(res_t, expected_t);
        Ok(())
    }

    fn join_fail(t0: Type, t1: Type, op: Operation) -> Result<()> {
        let context = create_unchecked_context()?;
        let graph = context.create_graph()?;
        let mut worker = create_type_inference_worker(context.clone());
        let i0 = graph.input(t0.clone())?;
        let i1 = graph.input(t1.clone())?;
        let o = graph.add_node(vec![i0, i1], vec![], op)?;
        let res_t = worker.process_node(o);
        assert!(res_t.is_err());
        Ok(())
    }

    fn join_helper(join_t: JoinType) -> Result<()> {
        join_worker(
            vec![
                create_column(NULL_HEADER, vec![1], BIT),
                create_column("ID", vec![1], UINT64),
                create_column("Country", vec![1], UINT8),
                create_column("Name", vec![128], BIT),
            ],
            1,
            vec![
                create_column(NULL_HEADER, vec![1], BIT),
                create_column("ID", vec![1], UINT64),
                create_column("Country", vec![1], UINT8),
                create_column("Name", vec![128], BIT),
            ],
            1,
            Operation::Join(
                join_t.clone(),
                HashMap::from([
                    ("ID".to_owned(), "ID".to_owned()),
                    ("Country".to_owned(), "Country".to_owned()),
                    ("Name".to_owned(), "Name".to_owned()),
                ]),
            ),
            vec![
                create_column(NULL_HEADER, vec![1], BIT),
                create_column("ID", vec![1], UINT64),
                create_column("Country", vec![1], UINT8),
                create_column("Name", vec![128], BIT),
            ],
        )?;

        join_worker(
            vec![
                create_column(NULL_HEADER, vec![1], BIT),
                create_column("ID", vec![1], UINT64),
                create_column("Country", vec![1], UINT8),
                create_column("First Name", vec![128], BIT),
            ],
            50,
            vec![
                create_column(NULL_HEADER, vec![1], BIT),
                create_column("UID", vec![1], UINT64),
                create_column("Origin", vec![1], UINT8),
                create_column("Second Name", vec![128], BIT),
            ],
            30,
            Operation::Join(
                join_t.clone(),
                HashMap::from([
                    ("ID".to_owned(), "UID".to_owned()),
                    ("Country".to_owned(), "Origin".to_owned()),
                ]),
            ),
            vec![
                create_column(NULL_HEADER, vec![1], BIT),
                create_column("ID", vec![1], UINT64),
                create_column("Country", vec![1], UINT8),
                create_column("First Name", vec![128], BIT),
                create_column("Second Name", vec![128], BIT),
            ],
        )?;
        join_worker(
            vec![
                create_column(NULL_HEADER, vec![1], BIT),
                create_column("ID1", vec![1], UINT64),
            ],
            50,
            vec![
                create_column(NULL_HEADER, vec![1], BIT),
                create_column("ID2", vec![1], UINT64),
            ],
            30,
            Operation::Join(
                join_t.clone(),
                HashMap::from([("ID1".to_owned(), "ID2".to_owned())]),
            ),
            vec![
                create_column(NULL_HEADER, vec![1], BIT),
                create_column("ID1", vec![1], UINT64),
            ],
        )?;

        join_fail(
            named_tuple_type(vec![
                (NULL_HEADER.to_owned(), array_type(vec![50], BIT)),
                ("ID1".to_owned(), array_type(vec![50], UINT64)),
            ]),
            named_tuple_type(vec![
                (NULL_HEADER.to_owned(), array_type(vec![30], BIT)),
                ("ID2".to_owned(), array_type(vec![30], UINT64)),
            ]),
            Operation::Join(join_t.clone(), HashMap::new()),
        )?;
        join_fail(
            named_tuple_type(vec![("ID1".to_owned(), array_type(vec![50], UINT64))]),
            named_tuple_type(vec![("ID2".to_owned(), array_type(vec![30], UINT64))]),
            Operation::Join(
                join_t.clone(),
                HashMap::from([("ID1".to_owned(), "ID2".to_owned())]),
            ),
        )?;
        join_fail(
            named_tuple_type(vec![
                (NULL_HEADER.to_owned(), array_type(vec![50], UINT64)),
                ("ID1".to_owned(), array_type(vec![50], UINT64)),
            ]),
            named_tuple_type(vec![
                (NULL_HEADER.to_owned(), array_type(vec![30], BIT)),
                ("ID2".to_owned(), array_type(vec![30], UINT64)),
            ]),
            Operation::Join(
                join_t.clone(),
                HashMap::from([("ID1".to_owned(), "ID2".to_owned())]),
            ),
        )?;
        join_fail(
            named_tuple_type(vec![
                (NULL_HEADER.to_owned(), array_type(vec![50], BIT)),
                ("ID1".to_owned(), array_type(vec![50], UINT64)),
            ]),
            named_tuple_type(vec![
                (NULL_HEADER.to_owned(), array_type(vec![30], UINT64)),
                ("ID2".to_owned(), array_type(vec![30], UINT64)),
            ]),
            Operation::Join(
                join_t.clone(),
                HashMap::from([("ID1".to_owned(), "ID2".to_owned())]),
            ),
        )?;
        join_fail(
            named_tuple_type(vec![
                (NULL_HEADER.to_owned(), array_type(vec![50, 1], BIT)),
                ("ID1".to_owned(), array_type(vec![50], UINT64)),
            ]),
            named_tuple_type(vec![
                (NULL_HEADER.to_owned(), array_type(vec![30], BIT)),
                ("ID2".to_owned(), array_type(vec![30], UINT64)),
            ]),
            Operation::Join(
                join_t.clone(),
                HashMap::from([("ID1".to_owned(), "ID2".to_owned())]),
            ),
        )?;
        join_fail(
            named_tuple_type(vec![
                (NULL_HEADER.to_owned(), array_type(vec![50], BIT)),
                ("ID1".to_owned(), array_type(vec![50], UINT64)),
            ]),
            named_tuple_type(vec![
                (NULL_HEADER.to_owned(), array_type(vec![30], BIT)),
                ("ID2".to_owned(), array_type(vec![30], UINT64)),
            ]),
            Operation::Join(
                join_t.clone(),
                HashMap::from([("ID1".to_owned(), "ID".to_owned())]),
            ),
        )?;
        join_fail(
            named_tuple_type(vec![
                (NULL_HEADER.to_owned(), array_type(vec![50], BIT)),
                ("ID1".to_owned(), array_type(vec![50], UINT64)),
            ]),
            named_tuple_type(vec![
                (NULL_HEADER.to_owned(), array_type(vec![30], BIT)),
                ("ID2".to_owned(), array_type(vec![30], UINT64)),
            ]),
            Operation::Join(
                join_t.clone(),
                HashMap::from([("ID".to_owned(), "ID2".to_owned())]),
            ),
        )?;
        join_fail(
            named_tuple_type(vec![
                (NULL_HEADER.to_owned(), array_type(vec![50], BIT)),
                ("ID1".to_owned(), array_type(vec![30], UINT64)),
            ]),
            named_tuple_type(vec![
                (NULL_HEADER.to_owned(), array_type(vec![30], BIT)),
                ("ID2".to_owned(), array_type(vec![30], UINT64)),
            ]),
            Operation::Join(
                join_t.clone(),
                HashMap::from([("ID1".to_owned(), "ID2".to_owned())]),
            ),
        )?;
        join_fail(
            named_tuple_type(vec![
                (NULL_HEADER.to_owned(), array_type(vec![50], BIT)),
                ("ID1".to_owned(), array_type(vec![50], UINT64)),
            ]),
            named_tuple_type(vec![
                (NULL_HEADER.to_owned(), array_type(vec![30], BIT)),
                ("ID2".to_owned(), array_type(vec![50], UINT64)),
            ]),
            Operation::Join(
                join_t.clone(),
                HashMap::from([("ID1".to_owned(), "ID2".to_owned())]),
            ),
        )?;
        join_fail(
            named_tuple_type(vec![
                (NULL_HEADER.to_owned(), array_type(vec![50], BIT)),
                ("ID1".to_owned(), array_type(vec![50], UINT32)),
            ]),
            named_tuple_type(vec![
                (NULL_HEADER.to_owned(), array_type(vec![30], BIT)),
                ("ID2".to_owned(), array_type(vec![30], UINT64)),
            ]),
            Operation::Join(
                join_t.clone(),
                HashMap::from([("ID1".to_owned(), "ID2".to_owned())]),
            ),
        )?;
        join_fail(
            named_tuple_type(vec![
                (NULL_HEADER.to_owned(), array_type(vec![50], BIT)),
                ("ID1".to_owned(), array_type(vec![50, 1], UINT64)),
            ]),
            named_tuple_type(vec![
                (NULL_HEADER.to_owned(), array_type(vec![50], BIT)),
                ("ID2".to_owned(), array_type(vec![50], UINT64)),
            ]),
            Operation::Join(
                join_t.clone(),
                HashMap::from([("ID1".to_owned(), "ID2".to_owned())]),
            ),
        )?;
        join_fail(
            named_tuple_type(vec![
                (NULL_HEADER.to_owned(), array_type(vec![30], BIT)),
                ("ID1".to_owned(), array_type(vec![30], UINT64)),
                ("Name".to_owned(), array_type(vec![30], UINT64)),
            ]),
            named_tuple_type(vec![
                (NULL_HEADER.to_owned(), array_type(vec![50], BIT)),
                ("ID2".to_owned(), array_type(vec![50], UINT64)),
                ("Name".to_owned(), array_type(vec![50], UINT64)),
            ]),
            Operation::Join(
                join_t.clone(),
                HashMap::from([("ID1".to_owned(), "ID2".to_owned())]),
            ),
        )?;
        join_fail(
            scalar_type(BIT),
            named_tuple_type(vec![
                (NULL_HEADER.to_owned(), array_type(vec![50], BIT)),
                ("ID2".to_owned(), array_type(vec![50], UINT64)),
                ("Name".to_owned(), array_type(vec![50], UINT64)),
            ]),
            Operation::Join(
                join_t.clone(),
                HashMap::from([("ID1".to_owned(), "ID2".to_owned())]),
            ),
        )?;
        join_fail(
            named_tuple_type(vec![
                ("ID1".to_owned(), array_type(vec![50], UINT64)),
                ("Name1".to_owned(), array_type(vec![50], UINT64)),
            ]),
            named_tuple_type(vec![
                ("ID2".to_owned(), array_type(vec![30], UINT64)),
                ("Name2".to_owned(), array_type(vec![30], UINT64)),
            ]),
            Operation::Join(
                join_t.clone(),
                HashMap::from([("ID1".to_owned(), "ID2".to_owned())]),
            ),
        )?;
        join_fail(
            named_tuple_type(vec![
                (NULL_HEADER.to_owned(), array_type(vec![30], BIT)),
                ("ID1".to_owned(), array_type(vec![30], UINT64)),
                ("Name1".to_owned(), scalar_type(UINT64)),
            ]),
            named_tuple_type(vec![
                (NULL_HEADER.to_owned(), array_type(vec![30], BIT)),
                ("ID2".to_owned(), array_type(vec![30], UINT64)),
                ("Name2".to_owned(), array_type(vec![30], UINT64)),
            ]),
            Operation::Join(
                join_t.clone(),
                HashMap::from([("ID1".to_owned(), "ID2".to_owned())]),
            ),
        )?;
        join_fail(
            named_tuple_type(vec![
                (NULL_HEADER.to_owned(), array_type(vec![30], BIT)),
                ("ID1".to_owned(), array_type(vec![30], UINT64)),
                ("Name1".to_owned(), array_type(vec![30], UINT64)),
            ]),
            named_tuple_type(vec![
                (NULL_HEADER.to_owned(), array_type(vec![50], BIT)),
                ("ID2".to_owned(), array_type(vec![50], UINT64)),
                ("Name2".to_owned(), array_type(vec![50], UINT64)),
            ]),
            Operation::Join(
                join_t.clone(),
                HashMap::from([(NULL_HEADER.to_owned(), NULL_HEADER.to_owned())]),
            ),
        )?;
        Ok(())
    }

    #[test]
    fn test_set_intersection() -> Result<()> {
        join_helper(JoinType::Inner)
    }

    #[test]
    fn test_left_join() -> Result<()> {
        join_helper(JoinType::Left)
    }

    #[test]
    fn test_union() -> Result<()> {
        join_helper(JoinType::Union)
    }

    #[test]
    fn test_full_join() -> Result<()> {
        join_helper(JoinType::Full)
    }

    fn test_concatenate_worker(ts: Vec<Type>, axis: u64, expected_t: Type) -> Result<()> {
        let context = create_unchecked_context()?;
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph()?;
        let mut nodes = vec![];
        for t in ts {
            nodes.push(graph.input(t)?);
        }
        let out = graph.concatenate(nodes, axis)?;
        let result = worker.process_node(out)?;
        assert_eq!(result, expected_t);
        Ok(())
    }

    fn test_concatenate_worker_fail(ts: Vec<Type>, axis: u64) -> Result<()> {
        let context = create_unchecked_context()?;
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph()?;
        let mut nodes = vec![];
        for t in ts {
            nodes.push(graph.input(t)?);
        }
        let out = graph.concatenate(nodes, axis)?;
        let e = worker.process_node(out);
        assert!(e.is_err());
        Ok(())
    }

    #[test]
    fn test_concatenate() {
        || -> Result<()> {
            test_concatenate_worker(
                vec![array_type(vec![1], BIT), array_type(vec![1], BIT)],
                0,
                array_type(vec![2], BIT),
            )?;
            test_concatenate_worker(
                vec![array_type(vec![5, 10], BIT), array_type(vec![1, 10], BIT)],
                0,
                array_type(vec![6, 10], BIT),
            )?;
            test_concatenate_worker(
                vec![
                    array_type(vec![3, 5, 10], INT32),
                    array_type(vec![3, 1, 10], INT32),
                ],
                1,
                array_type(vec![3, 6, 10], INT32),
            )?;
            test_concatenate_worker(
                vec![
                    array_type(vec![3, 10, 1], INT32),
                    array_type(vec![3, 10, 2], INT32),
                    array_type(vec![3, 10, 3], INT32),
                    array_type(vec![3, 10, 5], INT32),
                ],
                2,
                array_type(vec![3, 10, 11], INT32),
            )?;

            test_concatenate_worker_fail(
                vec![array_type(vec![1], BIT), array_type(vec![1], INT32)],
                0,
            )?;
            test_concatenate_worker_fail(
                vec![array_type(vec![1], BIT), array_type(vec![1], BIT)],
                1,
            )?;
            test_concatenate_worker_fail(vec![array_type(vec![1], BIT)], 0)?;
            test_concatenate_worker_fail(
                vec![
                    array_type(vec![10, 5, 4], INT32),
                    array_type(vec![10, 4, 3], INT32),
                ],
                1,
            )?;
            test_concatenate_worker_fail(
                vec![
                    array_type(vec![10, 5, 4], INT32),
                    array_type(vec![10, 4, 4], INT32),
                ],
                0,
            )?;
            test_concatenate_worker_fail(
                vec![array_type(vec![10, 5, 4], INT32), scalar_type(INT32)],
                0,
            )?;
            test_concatenate_worker_fail(
                vec![
                    array_type(vec![10, 5, 4], INT32),
                    array_type(vec![10, 5, 4], INT32),
                ],
                4,
            )?;
            test_concatenate_worker_fail(
                vec![
                    array_type(vec![10, 5, 4], INT32),
                    array_type(vec![5, 4], INT32),
                ],
                1,
            )?;
            test_concatenate_worker_fail(
                vec![
                    array_type(vec![10, 5, 4], INT32),
                    array_type(vec![5, 4], INT32),
                ],
                2,
            )?;

            Ok(())
        }()
        .unwrap();
    }

    fn test_gemm_worker(
        t0: Type,
        t1: Type,
        transpose0: bool,
        transpose1: bool,
        t2: Type,
    ) -> Result<()> {
        let context = create_unchecked_context()?;
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph()?;
        let i0 = graph.input(t0)?;
        let i1 = graph.input(t1)?;
        let out = graph.gemm(i0, i1, transpose0, transpose1)?;
        let t2_result = worker.process_node(out)?;
        assert_eq!(t2_result, t2);
        Ok(())
    }

    fn test_gemm_worker_fail(t0: Type, t1: Type, transpose0: bool, transpose1: bool) -> Result<()> {
        let context = create_unchecked_context().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let graph = context.create_graph().unwrap();
        let i0 = graph.input(t0).unwrap();
        let i1 = graph.input(t1).unwrap();
        let out = graph.gemm(i0, i1, transpose0, transpose1).unwrap();
        let e = worker.process_node(out);
        assert!(e.is_err());
        Ok(())
    }

    #[test]
    fn test_gemm() {
        || -> Result<()> {
            test_gemm_worker(
                array_type(vec![10, 20], INT32),
                array_type(vec![20, 30], INT32),
                false,
                false,
                array_type(vec![10, 30], INT32),
            )?;
            test_gemm_worker(
                array_type(vec![20, 10], INT32),
                array_type(vec![20, 30], INT32),
                true,
                false,
                array_type(vec![10, 30], INT32),
            )?;
            test_gemm_worker(
                array_type(vec![20, 10], INT32),
                array_type(vec![30, 20], INT32),
                true,
                true,
                array_type(vec![10, 30], INT32),
            )?;
            test_gemm_worker(
                array_type(vec![10, 20], INT32),
                array_type(vec![30, 20], INT32),
                false,
                true,
                array_type(vec![10, 30], INT32),
            )?;
            test_gemm_worker(
                array_type(vec![10, 20, 50], INT32),
                array_type(vec![10, 50, 70], INT32),
                false,
                false,
                array_type(vec![10, 20, 70], INT32),
            )?;
            test_gemm_worker(
                array_type(vec![10, 50, 20], INT32),
                array_type(vec![10, 50, 70], INT32),
                true,
                false,
                array_type(vec![10, 20, 70], INT32),
            )?;
            test_gemm_worker(
                array_type(vec![10, 20, 50], INT32),
                array_type(vec![10, 70, 50], INT32),
                false,
                true,
                array_type(vec![10, 20, 70], INT32),
            )?;
            test_gemm_worker(
                array_type(vec![10, 50, 20], INT32),
                array_type(vec![10, 70, 50], INT32),
                true,
                true,
                array_type(vec![10, 20, 70], INT32),
            )?;
            test_gemm_worker(
                array_type(vec![10, 20, 30], INT32),
                array_type(vec![30, 70], INT32),
                false,
                false,
                array_type(vec![10, 20, 70], INT32),
            )?;
            test_gemm_worker(
                array_type(vec![10, 30, 20], INT32),
                array_type(vec![30, 70], INT32),
                true,
                false,
                array_type(vec![10, 20, 70], INT32),
            )?;
            test_gemm_worker(
                array_type(vec![10, 20, 30], INT32),
                array_type(vec![70, 30], INT32),
                false,
                true,
                array_type(vec![10, 20, 70], INT32),
            )?;
            test_gemm_worker(
                array_type(vec![10, 30, 20], INT32),
                array_type(vec![70, 30], INT32),
                true,
                true,
                array_type(vec![10, 20, 70], INT32),
            )?;
            test_gemm_worker(
                array_type(vec![20, 30], INT32),
                array_type(vec![10, 30, 70], INT32),
                false,
                false,
                array_type(vec![10, 20, 70], INT32),
            )?;
            test_gemm_worker(
                array_type(vec![30, 20], INT32),
                array_type(vec![10, 30, 70], INT32),
                true,
                false,
                array_type(vec![10, 20, 70], INT32),
            )?;
            test_gemm_worker(
                array_type(vec![20, 30], INT32),
                array_type(vec![10, 70, 30], INT32),
                false,
                true,
                array_type(vec![10, 20, 70], INT32),
            )?;
            test_gemm_worker(
                array_type(vec![30, 20], INT32),
                array_type(vec![10, 70, 30], INT32),
                true,
                true,
                array_type(vec![10, 20, 70], INT32),
            )?;
            test_gemm_worker(
                array_type(vec![1, 3, 4, 20, 30], INT32),
                array_type(vec![5, 3, 1, 30, 70], INT32),
                false,
                false,
                array_type(vec![5, 3, 4, 20, 70], INT32),
            )?;
            test_gemm_worker(
                array_type(vec![1, 3, 4, 30, 20], INT32),
                array_type(vec![5, 3, 1, 30, 70], INT32),
                true,
                false,
                array_type(vec![5, 3, 4, 20, 70], INT32),
            )?;
            test_gemm_worker(
                array_type(vec![1, 3, 4, 20, 30], INT32),
                array_type(vec![5, 3, 1, 70, 30], INT32),
                false,
                true,
                array_type(vec![5, 3, 4, 20, 70], INT32),
            )?;
            test_gemm_worker(
                array_type(vec![1, 3, 4, 30, 20], INT32),
                array_type(vec![5, 3, 1, 70, 30], INT32),
                true,
                true,
                array_type(vec![5, 3, 4, 20, 70], INT32),
            )?;

            test_gemm_worker_fail(
                array_type(vec![10], INT32),
                array_type(vec![10], INT32),
                true,
                true,
            )?;
            test_gemm_worker_fail(
                array_type(vec![10, 20, 50], INT32),
                array_type(vec![50], INT32),
                false,
                false,
            )?;
            test_gemm_worker_fail(scalar_type(INT32), scalar_type(BIT), false, false)?;
            test_gemm_worker_fail(
                array_type(vec![50], INT32),
                array_type(vec![100], INT32),
                false,
                false,
            )?;
            test_gemm_worker_fail(
                array_type(vec![10, 50], INT32),
                array_type(vec![100, 3], INT32),
                false,
                false,
            )?;
            test_gemm_worker_fail(
                array_type(vec![10, 50], INT32),
                array_type(vec![100], INT32),
                false,
                false,
            )?;
            test_gemm_worker_fail(
                array_type(vec![10, 20], INT32),
                scalar_type(INT32),
                false,
                false,
            )?;
            test_gemm_worker_fail(
                scalar_type(INT32),
                array_type(vec![10, 20], INT32),
                false,
                false,
            )?;
            test_gemm_worker_fail(scalar_type(INT32), scalar_type(INT32), false, false)?;
            test_gemm_worker_fail(
                array_type(vec![10, 50, 100], INT32),
                array_type(vec![100, 30, 50], INT32),
                false,
                false,
            )?;
            test_gemm_worker_fail(
                array_type(vec![10, 20, 50], INT32),
                array_type(vec![30, 50, 70], INT32),
                false,
                false,
            )?;
            test_gemm_worker_fail(
                array_type(vec![10, 20, 50], INT32),
                array_type(vec![10, 50, 70], INT32),
                true,
                false,
            )?;
            test_gemm_worker_fail(
                array_type(vec![10, 20, 50], INT32),
                array_type(vec![10, 50, 70], INT32),
                true,
                true,
            )?;
            test_gemm_worker_fail(
                array_type(vec![10, 20, 50], INT32),
                array_type(vec![10, 50, 70], INT32),
                false,
                true,
            )?;

            Ok(())
        }()
        .unwrap();
    }

    fn test_print_worker(t: Type) {
        let context = create_unchecked_context().unwrap();
        let graph = context.create_graph().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let i = graph.input(t.clone()).unwrap();
        let o = graph.print("I:".into(), i).unwrap();
        let ot = worker.process_node(o).unwrap();
        assert_eq!(ot, t);
    }

    #[test]
    fn test_print() {
        test_print_worker(array_type(vec![10, 3, 8], BIT));
        test_print_worker(scalar_type(INT32));
        test_print_worker(tuple_type(vec![]));
    }

    fn test_assert_worker(t0: Type, t1: Type) {
        let context = create_unchecked_context().unwrap();
        let graph = context.create_graph().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let i0 = graph.input(t0).unwrap();
        let i1 = graph.input(t1.clone()).unwrap();
        let o = graph.assert("i0 is true".into(), i0, i1).unwrap();
        let t = worker.process_node(o).unwrap();
        assert_eq!(t, t1);
    }

    fn test_assert_worker_fail(t0: Type, t1: Type) {
        let context = create_unchecked_context().unwrap();
        let graph = context.create_graph().unwrap();
        let mut worker = create_type_inference_worker(context.clone());
        let i0 = graph.input(t0).unwrap();
        let i1 = graph.input(t1.clone()).unwrap();
        let o = graph.assert("i0 is true".into(), i0, i1).unwrap();
        let res = worker.process_node(o);
        assert!(res.is_err());
    }

    #[test]
    fn test_assert() {
        test_assert_worker(scalar_type(BIT), array_type(vec![10, 3, 8], BIT));
        test_assert_worker(scalar_type(BIT), scalar_type(INT32));
        test_assert_worker(scalar_type(BIT), tuple_type(vec![]));
        test_assert_worker_fail(scalar_type(INT32), tuple_type(vec![]));
        test_assert_worker_fail(array_type(vec![10], BIT), tuple_type(vec![]));
    }

    fn expected_sharding(headers_types: Vec<(&str, Type)>, shard_config: &ShardConfig) -> Type {
        let mut shard_data_vec = vec![];
        for (h, t) in headers_types {
            let mut shape = t.get_shape();
            shape[0] = shard_config.shard_size;
            shard_data_vec.push((h.to_owned(), array_type(shape, t.get_scalar_type())));
        }
        let shard_data = named_tuple_type(shard_data_vec);
        let shard_mask = array_type(vec![shard_config.shard_size], BIT);
        let shard = tuple_type(vec![shard_mask, shard_data]);
        tuple_type(vec![shard; shard_config.num_shards as usize])
    }

    fn test_shard_worker(
        headers_types: Vec<(&str, Type)>,
        shard_config: ShardConfig,
    ) -> Result<()> {
        let context = create_unchecked_context()?;
        let graph = context.create_graph()?;
        let mut worker = create_type_inference_worker(context.clone());
        let i = graph.input(named_tuple_type(
            headers_types
                .iter()
                .map(|(h, t)| ((*h).to_owned(), t.clone()))
                .collect(),
        ))?;
        let o = graph.shard(i, shard_config.clone())?;
        let t = worker.process_node(o)?;

        let expected_t = expected_sharding(headers_types, &shard_config);
        assert_eq!(t, expected_t);

        Ok(())
    }

    fn test_shard_fail(input_t: Type, shard_config: ShardConfig) -> Result<()> {
        let context = create_unchecked_context()?;
        let graph = context.create_graph()?;
        let mut worker = create_type_inference_worker(context.clone());
        let i = graph.input(input_t)?;
        let o = graph.shard(i, shard_config)?;
        let t = worker.process_node(o);
        assert!(t.is_err());

        Ok(())
    }

    #[test]
    fn test_shard() -> Result<()> {
        test_shard_worker(
            vec![("ID", array_type(vec![10], UINT64))],
            ShardConfig {
                num_shards: 2,
                shard_size: 7,
                shard_headers: vec!["ID".to_owned()],
            },
        )?;

        test_shard_worker(
            vec![
                ("ID", array_type(vec![10], UINT64)),
                ("Income per month", array_type(vec![10, 12], UINT64)),
            ],
            ShardConfig {
                num_shards: 3,
                shard_size: 5,
                shard_headers: vec!["Income per month".to_owned()],
            },
        )?;

        test_shard_worker(
            vec![
                ("ID", array_type(vec![10], UINT64)),
                ("Income per month", array_type(vec![10, 12], UINT64)),
            ],
            ShardConfig {
                num_shards: 3,
                shard_size: 5,
                shard_headers: vec!["ID".to_owned(), "Income per month".to_owned()],
            },
        )?;

        test_shard_worker(
            vec![
                ("ID", array_type(vec![1], UINT64)),
                ("Income per month", array_type(vec![1, 12], UINT64)),
                ("Outcome", array_type(vec![1], UINT32)),
            ],
            ShardConfig {
                num_shards: 3,
                shard_size: 1,
                shard_headers: vec!["ID".to_owned(), "Income per month".to_owned()],
            },
        )?;

        let shard_config = ShardConfig {
            num_shards: 4,
            shard_size: 10,
            shard_headers: vec!["ID".to_owned()],
        };
        test_shard_fail(array_type(vec![5], BIT), shard_config.clone())?;
        test_shard_fail(
            named_tuple_type(vec![("Income".to_owned(), array_type(vec![10], UINT64))]),
            shard_config.clone(),
        )?;
        test_shard_fail(
            named_tuple_type(vec![
                ("Income".to_owned(), array_type(vec![10], UINT64)),
                ("ID".to_owned(), array_type(vec![8], UINT64)),
            ]),
            shard_config.clone(),
        )?;
        test_shard_fail(
            named_tuple_type(vec![
                ("Income".to_owned(), array_type(vec![80], UINT64)),
                ("ID".to_owned(), array_type(vec![80], UINT64)),
            ]),
            shard_config.clone(),
        )?;
        test_shard_fail(
            named_tuple_type(vec![
                ("Income".to_owned(), array_type(vec![20], UINT64)),
                ("ID".to_owned(), array_type(vec![20], UINT64)),
            ]),
            ShardConfig {
                num_shards: 4,
                shard_size: 10,
                shard_headers: vec![],
            },
        )?;
        test_shard_fail(
            named_tuple_type(vec![
                ("Income".to_owned(), array_type(vec![20], UINT64)),
                ("ID".to_owned(), array_type(vec![20], UINT64)),
            ]),
            ShardConfig {
                num_shards: 4,
                shard_size: 10,
                shard_headers: vec!["ID".to_owned(), "ID".to_owned()],
            },
        )?;
        test_shard_fail(
            named_tuple_type(vec![
                ("Income".to_owned(), array_type(vec![20], UINT64)),
                ("ID".to_owned(), array_type(vec![20], UINT64)),
            ]),
            ShardConfig {
                num_shards: 800,
                shard_size: 10,
                shard_headers: vec!["ID".to_owned()],
            },
        )?;

        Ok(())
    }
}
