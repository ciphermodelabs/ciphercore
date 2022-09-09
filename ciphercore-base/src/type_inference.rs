use crate::broadcast::{broadcast_arrays, broadcast_shapes};
use crate::custom_ops::Instantiation;
use crate::data_types::{
    array_type, is_valid_shape, named_tuple_type, scalar_size_in_bits, scalar_type, tuple_type,
    vector_type, ScalarType, Type, BIT, UINT32, UINT64,
};
use crate::errors::Result;
use crate::graphs::{create_context, Context, Node, Operation, WeakContext};
use crate::slices::get_slice_shape;
use std::collections::{HashMap, HashSet};

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
            "The first argument of mixed multiply is not a scalar or an array"
        ));
    }
    if !t1.is_scalar() && !t1.is_array() {
        return Err(runtime_error!(
            "The second argument of mixed multiply is not a scalar or an array"
        ));
    }

    if t0.get_scalar_type() == BIT {
        return Err(runtime_error!(
            "The scalar type of the first argument shouldn't be BIT"
        ));
    }
    if t1.get_scalar_type() != BIT {
        return Err(runtime_error!(
            "The scalar type of the second argument must be BIT"
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
            "The first argument of dot is not a scalar or an array"
        ));
    }
    if !t1.is_scalar() && !t1.is_array() {
        return Err(runtime_error!(
            "The second argument of dot is not a scalar or an array"
        ));
    }
    if t0.get_scalar_type() != t1.get_scalar_type() {
        return Err(runtime_error!("Incompatible scalar types"));
    }
    let st = t0.get_scalar_type();
    if t0.is_array() && t1.is_array() {
        let s0 = t0.get_shape();
        let s1 = t1.get_shape();
        if s0.len() == 1 && s1.len() == 1 {
            if s0[0] != s1[0] {
                return Err(runtime_error!("Dot with incompatible dimensions"));
            }
            Ok(scalar_type(st))
        } else if s1.len() == 1 {
            if s0[s0.len() - 1] != s1[0] {
                Err(runtime_error!("Dot with incompatible dimensions"))
            } else {
                let mut sr = s0.clone();
                sr.remove(s0.len() - 1);
                Ok(array_type(sr, st))
            }
        } else if s0[s0.len() - 1] != s1[s1.len() - 2] {
            Err(runtime_error!("Dot with incompatible dimensions"))
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
            "The first argument of matmul is not an array"
        ));
    }
    if !t1.is_array() {
        return Err(runtime_error!(
            "The second argument of matmul is not an array"
        ));
    }
    if t0.get_scalar_type() != t1.get_scalar_type() {
        return Err(runtime_error!("Incompatible scalar types"));
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
        return Err(runtime_error!("Matmul with incompatible dimensions"));
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

pub(super) fn a2b_type_inference(original_type: Type) -> Result<Type> {
    if !original_type.is_scalar() && !original_type.is_array() {
        return Err(runtime_error!(
            "Invalid type for A2B: can only be array or scalar"
        ));
    }
    let st = original_type.get_scalar_type();
    if st == BIT {
        return Err(runtime_error!("A2B can't be applied to bits"));
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
        return Err(runtime_error!("Invalid type"));
    }
    if !t.is_array() {
        return Err(runtime_error!("Trying to B2A non-array"));
    }
    let mut shape = t.get_shape();
    let array_st = t.get_scalar_type();
    if array_st != BIT {
        return Err(runtime_error!("Trying to B2A from non-bits"));
    }
    if st == BIT {
        return Err(runtime_error!("Trying to B2A into bits"));
    }
    if shape[shape.len() - 1] != scalar_size_in_bits(st.clone()) {
        return Err(runtime_error!("Invalid scalar type for B2A"));
    }
    if shape.len() == 1 {
        Ok(scalar_type(st))
    } else {
        shape.pop();
        Ok(array_type(shape, st))
    }
}

/// Returns Some(n) if a given operation requires n node dependencies.
/// None means the number can be variable.
fn get_number_of_node_dependencies(operation: Operation) -> Option<u64> {
    match operation {
        Operation::Input(_) | Operation::Random(_) | Operation::Constant(_, _) => Some(0),
        Operation::Truncate(_)
        | Operation::Sum(_)
        | Operation::PermuteAxes(_)
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
        | Operation::VectorToArray => Some(1),
        Operation::Add
        | Operation::Subtract
        | Operation::Multiply
        | Operation::MixedMultiply
        | Operation::Dot
        | Operation::Matmul
        | Operation::VectorGet
        | Operation::Gather(_)
        | Operation::Iterate
        | Operation::CuckooHash => Some(2),
        Operation::Stack(_)
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

    pub fn process_node(&mut self, node: Node) -> Result<Type> {
        if self.context.upgrade() != node.get_graph().get_context() {
            return Err(runtime_error!(
                "Can't process a node from a different context"
            ));
        }
        let node_global_id = get_node_global_id(node.clone());
        if self.cached_results.contains_key(&node_global_id) {
            return Ok(self
                .cached_results
                .get(&node_global_id)
                .expect("Should not be here")
                .clone());
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
                    return Err(runtime_error!(
                        "Input with an invalid type: {:?}",
                        input_type
                    ));
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
            Operation::Truncate(d) => {
                let t = node_dependencies_types[0].clone();
                if d == 0 {
                    return Err(runtime_error!("Can't divide by zero"));
                }
                if !t.is_array() && !t.is_scalar() {
                    return Err(runtime_error!("Can't truncate this type"));
                }
                if t.get_scalar_type().get_signed() && d > i64::MAX as u64 {
                    return Err(runtime_error!("Scale for truncation is too large"));
                }
                self.register_result(node, t.clone())?;
                Ok(t)
            }
            Operation::Sum(s) => {
                let t = node_dependencies_types[0].clone();
                if !t.is_array() {
                    return Err(runtime_error!("Can't sum this type"));
                }
                let os = t.get_shape();
                let mut tmp = s.clone();
                tmp.sort_unstable();
                tmp.dedup();
                if tmp.len() < s.len() {
                    return Err(runtime_error!("Non-unique axes"));
                }
                let mut set: HashSet<u64> = HashSet::new();
                for x in s {
                    if x >= os.len() as u64 {
                        return Err(runtime_error!("Invalid axis"));
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
                    return Err(runtime_error!("Can't permute_axes this type"));
                }
                let os = t.get_shape();
                let mut tmp = s.clone();
                tmp.sort_unstable();
                tmp.dedup();
                if tmp.len() < s.len() {
                    return Err(runtime_error!("Non-unique axes"));
                }
                for x in &s {
                    if *x >= os.len() as u64 {
                        return Err(runtime_error!("Invalid axes"));
                    }
                }
                if s.len() != os.len() {
                    return Err(runtime_error!("Not a permutation"));
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
            Operation::Get(s) => {
                let t = node_dependencies_types[0].clone();
                if !t.is_array() {
                    return Err(runtime_error!("Can't run get on this type"));
                }
                let os = t.get_shape();
                if s.len() > os.len() {
                    return Err(runtime_error!("Too long index"));
                }
                for i in 0..s.len() {
                    if s[i] >= os[i] {
                        return Err(runtime_error!("Out of bounds"));
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
                    return Err(runtime_error!("Can't run get_slice on this type"));
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
                    return Err(runtime_error!("Incompatible types for reshape"));
                }
                let v1 = flatten_type(old_type);
                let v2 = flatten_type(new_type.clone());
                for i in 0..v1.len() {
                    if !can_atomic_reshape(v1[i].clone(), v2[i].clone()) {
                        return Err(runtime_error!("Incompatible types for reshape"));
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
            Operation::PRF(_, ot) => {
                let t = node_dependencies_types[0].clone();
                if !t.is_array() {
                    return Err(runtime_error!("PRF key must be an array"));
                }
                let s = t.get_shape();
                let st = t.get_scalar_type();
                if s.len() != 1 || s[0] != 128 || st != BIT {
                    return Err(runtime_error!("PRF key must consist of 128 bits"));
                }
                self.register_result(node, ot.clone())?;
                Ok(ot)
            }
            Operation::Stack(outer_shape) => {
                if !is_valid_shape(outer_shape.clone()) {
                    return Err(runtime_error!("Invalid outer shape"));
                }
                let mut pr = 1;
                for x in &outer_shape {
                    pr *= *x;
                }
                if node_dependencies.len() as u64 != pr {
                    return Err(runtime_error!("Stack with a wrong number of arguments"));
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
            Operation::Constant(t, ref value) => {
                if !value.check_type(t.clone())? {
                    return Err(runtime_error!("Invalid constant type"));
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
                    return Err(runtime_error!("Invalid number of fields provided"));
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
                        return Err(runtime_error!("Vector element type mismatch"));
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
                            return Err(runtime_error!("Index is out of bounds"));
                        }
                        (*fields[field_id as usize]).clone()
                    }
                    Type::NamedTuple(fields) => {
                        if field_id >= fields.len() as u64 {
                            return Err(runtime_error!("Index is out of bounds"));
                        }
                        let (_, t) = fields[field_id as usize].clone();
                        (*t).clone()
                    }
                    _ => {
                        return Err(runtime_error!("Can't TupleGet from this type"));
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
                        return Err(runtime_error!("Can't TupleGet from this type"));
                    }
                };
                Err(runtime_error!("Invalid field name"))
            }
            Operation::VectorGet => {
                let index_type = node_dependencies_types[1].clone();
                if index_type != scalar_type(UINT64) && index_type != scalar_type(UINT32) {
                    return Err(runtime_error!("Vector index must be an UINT64 or UINT32."));
                }
                let vector_type = node_dependencies_types[0].clone();
                let result = match vector_type {
                    Type::Vector(_, inner_type) => (*inner_type).clone(),
                    _ => {
                        return Err(runtime_error!("VectorGet can only be applied to vectors"));
                    }
                };
                self.register_result(node, result.clone())?;
                Ok(result)
            }
            Operation::Zip => {
                if node_dependencies.len() < 2 {
                    return Err(runtime_error!("Zip with a wrong number of arguments"));
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
                                        return Err(runtime_error!("Zip of uneven lengths"));
                                    }
                                }
                                None => {
                                    length = Some(*l);
                                }
                            }
                            element_types.push((*et).clone());
                        }
                        _ => {
                            return Err(runtime_error!("An argument of zip is not a vector"));
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
                    return Err(runtime_error!("Invalid number of arguments in Call"));
                }
                for i in 0..node_dependencies.len() {
                    if input_types[i] != node_dependencies_types[i] {
                        return Err(runtime_error!("Type mismatch for argument {}", i));
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
                        "Iterate graph must have two inputs: state and current input"
                    ));
                }
                let state_type = input_types[0].clone();
                let input_sequence_type = input_types[1].clone();
                let output_type = self.process_node(graph.get_output_node()?)?;
                match output_type {
                    Type::Tuple(element_types) => {
                        if element_types.len() != 2 {
                            return Err(runtime_error!(
                                "Iterate graph must output a tuple of two elements as an output"
                            ));
                        }
                        if *element_types[0] != state_type {
                            return Err(runtime_error!("State type mismatch"));
                        }
                        let output_sequence_type = (*element_types[1]).clone();
                        let t0 = node_dependencies_types[0].clone();
                        let t1 = node_dependencies_types[1].clone();
                        if t0 != state_type {
                            return Err(runtime_error!("Invalid state type"));
                        }
                        match t1 {
                            Type::Vector(len, element_type) => {
                                if *element_type != input_sequence_type {
                                    return Err(runtime_error!("Invalid sequence type"));
                                }
                                let result = tuple_type(vec![
                                    state_type,
                                    vector_type(len, output_sequence_type),
                                ]);
                                self.register_result(node, result.clone())?;
                                Ok(result)
                            }
                            _ => Err(runtime_error!("Invalid sequence type")),
                        }
                    }
                    _ => Err(runtime_error!("Iterate graph must output a tuple")),
                }
            }
            Operation::ArrayToVector => {
                let t = node_dependencies_types[0].clone();
                if !t.is_array() {
                    return Err(runtime_error!("ArrayToVector applied to a non-array"));
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
                            "VectorToArray can be only applied to a vector of scalars or arrays"
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
                        "VectorToArray can't be applied to a non-vector"
                    ))
                }
            }
            Operation::Gather(axis) => {
                let input_t = node_dependencies_types[0].clone();
                if !input_t.is_array() {
                    return Err(runtime_error!("Take can be only applied to an array"));
                }
                let indices_t = node_dependencies_types[1].clone();
                // TODO: support UINT32
                if !matches!(indices_t, Type::Array(_, UINT64)) {
                    return Err(runtime_error!("Indices must be an array of UINT64"));
                }
                let input_shape = input_t.get_shape();
                if axis >= input_shape.len() as u64 {
                    return Err(runtime_error!(
                        "Invalid axis. The axis index should be smaller than {}",
                        input_shape.len()
                    ));
                }
                let indices_shape = indices_t.get_shape();
                let indices_size = indices_shape.iter().product::<u64>();
                if indices_size > input_shape[axis as usize] {
                    return Err(runtime_error!(
                        "Number of indices is too big. At most {} elements can be extracted.",
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
                        "CuckooHash can't be applied to a non-binary arrays"
                    ));
                }
                let input_shape = input_t.get_shape();
                if input_shape.len() < 2 {
                    return Err(runtime_error!(
                        "Input shape must have at least 2 dimensions"
                    ));
                }
                if !matches!(hash_t, Type::Array(_, BIT)) {
                    return Err(runtime_error!(
                        "CuckooHash needs a binary array as a hash matrix"
                    ));
                }
                let hash_shape = hash_t.get_shape();
                if hash_shape.len() != 3 {
                    return Err(runtime_error!("Hash array should have 3 dimensions"));
                }
                if hash_shape[0] < 3 {
                    return Err(runtime_error!(
                        "At least 3 hash matrices should be provided"
                    ));
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
                // For each subarray, the output hash map contains indices of this array
                let mut output_shape = input_shape[0..input_shape.len() - 2].to_vec();
                let hash_map_size = 1 << hash_shape[1];
                output_shape.push(hash_map_size);
                let result = array_type(output_shape, UINT64);
                self.register_result(node, result.clone())?;
                Ok(result)
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
    use crate::data_types::{create_scalar_type, ArrayShape, Type, BIT, INT32, INT8, UINT32};
    use crate::data_values::Value;
    use crate::graphs::{create_unchecked_context, Graph, Slice, SliceElement};

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
}
