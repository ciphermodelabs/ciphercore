use crate::bytes::{add_vectors_u128, subtract_vectors_u128, vec_to_bytes, vec_u128_from_bytes};
use crate::data_types::{
    array_type, get_size_in_bits, get_types_vector, named_tuple_type, tuple_type, vector_type,
    ScalarType, Type,
};
use crate::data_values;
use crate::data_values::Value;
use crate::errors::Result;
use crate::random::PRNG;
use crate::typed_value_operations::{
    extend_helper, get_helper, get_sub_vector_helper, insert_helper, pop_helper, push_helper,
    remove_helper, FromVectorMode, ToNdarray, TypedValueArrayOperations, TypedValueOperations,
};

use std::ops::Not;

#[cfg(feature = "py-binding")]
use crate::data_types::PyBindingType;
#[cfg(feature = "py-binding")]
use crate::data_values::PyBindingValue;
#[cfg(feature = "py-binding")]
use pywrapper_macro::struct_wrapper;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::convert::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::describe::WasmDescribe;

/// A structure that stores pointer to a value and its type that corresponds to an input, output or an intermediate result of
/// a computation.
///
/// # Rust crates
///
/// [Clone] trait duplicates the pointer, not the underlying value (see [Value::deep_clone] for deep cloning).
///
/// [PartialEq] trait performs the deep recursive comparison.
#[derive(Clone, PartialEq, Eq)]
#[cfg_attr(feature = "py-binding", struct_wrapper)]
pub struct TypedValue {
    pub value: Value,
    pub t: Type,
    pub name: Option<String>,
}

impl std::fmt::Debug for TypedValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match serde_json::to_string(self) {
            Ok(json) => f.write_str(&json),
            Err(err) => f.write_fmt(format_args!(
                "Failed to convert TypedValue to json: {err:?}"
            )),
        }
    }
}

#[cfg(feature = "py-binding")]
#[pyo3::pymethods]
impl PyBindingTypedValue {
    #[staticmethod]
    fn from_str(value: String) -> pyo3::PyResult<Self> {
        Ok(PyBindingTypedValue {
            inner: serde_json::from_str::<TypedValue>(&value)
                .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?,
        })
    }

    /// Creates a typed value from a given type and value.
    /// Checks that the value is a valid for the given type.
    /// Note: check might be not sufficient.
    ///
    /// # Arguments
    ///
    /// `t` - the type
    /// `value` - the value
    ///
    /// # Returns
    ///
    /// New typed value constructed from the given type and value.
    #[staticmethod]
    fn from_type_and_value(
        t: &PyBindingType,
        value: &PyBindingValue,
    ) -> pyo3::PyResult<PyBindingTypedValue> {
        Ok(PyBindingTypedValue {
            inner: TypedValue::new(t.inner.clone(), value.inner.clone())?,
        })
    }

    fn get_local_shares_for_each_party(&self) -> pyo3::PyResult<Vec<Self>> {
        let mut prng = PRNG::new(None)?;
        Ok(self
            .inner
            .get_local_shares_for_each_party(&mut prng)?
            .into_iter()
            .map(|x| PyBindingTypedValue { inner: x })
            .collect())
    }
    fn get_type(&self) -> PyBindingType {
        PyBindingType {
            inner: self.inner.t.clone(),
        }
    }
    fn get_value(&self) -> PyBindingValue {
        PyBindingValue {
            inner: self.inner.value.clone(),
        }
    }
    fn __str__(&self) -> pyo3::PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))
    }
    fn __repr__(&self) -> pyo3::PyResult<String> {
        self.__str__()
    }
}

// There are several blockers for proper WASM bindings:
// * All wasm-bound classes need to implement `Copy`;
// * `Type` in `TypedValue` is a Rust enum, but only plain C-style enums are supported by wasm-bindgen.
// To circumvent this, we use a simple (but not very functional) binding - JSON representation of the `TypedValue`.
// Note that this can have consequences for efficiency if we pass large `TypedValue`s between Rust and JS code.
#[cfg(target_arch = "wasm32")]
impl WasmDescribe for TypedValue {
    fn describe() {
        <String as wasm_bindgen::describe::WasmDescribe>::describe()
    }
}

#[cfg(target_arch = "wasm32")]
impl FromWasmAbi for TypedValue {
    type Abi = <String as FromWasmAbi>::Abi;

    unsafe fn from_abi(js: Self::Abi) -> Self {
        let json: String = FromWasmAbi::from_abi(js);
        serde_json::from_str(&json).unwrap()
    }
}

#[cfg(target_arch = "wasm32")]
impl IntoWasmAbi for TypedValue {
    type Abi = <String as FromWasmAbi>::Abi;

    fn into_abi(self) -> Self::Abi {
        serde_json::to_string(&self).unwrap().into_abi()
    }
}

impl TypedValueOperations<TypedValue> for TypedValue {
    /// Returns Type of `self`
    ///
    /// # Returns
    ///
    /// Type of the TypedValue
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ciphercore_base::typed_value_operations::TypedValueOperations;
    /// # use ciphercore_base::data_types::{scalar_type,INT32};
    /// let tv = TypedValue::from_scalar(1, INT32).unwrap();
    /// assert!(tv.get_type().eq(&scalar_type(INT32)));
    /// ```
    fn get_type(&self) -> Type {
        self.t.clone()
    }
    /// Checks equality of `self` with another type value.
    ///
    /// # Returns
    ///
    /// True if `self` == `other` false otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ciphercore_base::typed_value_operations::TypedValueOperations;
    /// # use ciphercore_base::data_types::{scalar_type,INT32};
    /// let tv1 = TypedValue::from_scalar(1, INT32).unwrap();
    /// let tv2 = TypedValue::from_scalar(1, INT32).unwrap();
    /// assert!(tv1.is_equal(&tv2).unwrap());
    /// ```
    fn is_equal(&self, other: &Self) -> Result<bool> {
        const TWO: u8 = 2;
        let t = self.t.clone();
        if other.t != t {
            return Ok(false);
        }
        let type_size_in_bits = get_size_in_bits(t.clone())?;
        match t {
            Type::Scalar(_) | Type::Array(_, _) => {
                let self_bytes = self
                    .value
                    .access_bytes(|ref_bytes| Ok(ref_bytes.to_vec()))?;
                let other_bytes = other
                    .value
                    .access_bytes(|ref_bytes| Ok(ref_bytes.to_vec()))?;
                let value_size_in_bytes = self_bytes.len();
                let r: u32 = (type_size_in_bits % 8).try_into()?;
                let mut complete_bytes = value_size_in_bytes;
                if r != 0 {
                    complete_bytes -= 1;
                }
                for i in 0..complete_bytes {
                    if self_bytes[i] != other_bytes[i] {
                        return Ok(false);
                    }
                }
                if self_bytes[value_size_in_bytes - 1] % TWO.pow(r)
                    != other_bytes[value_size_in_bytes - 1] % TWO.pow(r)
                {
                    return Ok(false);
                }
                Ok(true)
            }
            Type::Vector(_, _) | Type::Tuple(_) | Type::NamedTuple(_) => {
                let types_vector = get_types_vector(t)?;
                let self_value_vector = self.value.access_vector(|vec| Ok(vec.clone()))?;
                let other_value_vector = other.value.access_vector(|vec| Ok(vec.clone()))?;
                // we are sure that types_vector.len() == self_value_vector.len() == other_value_vector.len()
                // because in generating TypedValue we recursively call check_type for all values.
                for i in 0..types_vector.len() {
                    let typed_value1 =
                        TypedValue::new((*types_vector[i]).clone(), self_value_vector[i].clone())?;
                    let typed_value2 =
                        TypedValue::new((*types_vector[i]).clone(), other_value_vector[i].clone())?;
                    if !typed_value1.is_equal(&typed_value2)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
        }
    }
    /// Converts `self` to a vector of typed values or return an error if `self` is a byte vector.
    ///
    /// # Returns
    ///
    /// Extracted vector of values
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ciphercore_base::typed_value_operations::{TypedValueOperations,FromVectorMode};
    /// # use ciphercore_base::data_types::{scalar_type,INT32};
    ///
    /// let v = vec![TypedValue::from_scalar(1,INT32).unwrap(),
    ///     TypedValue::from_scalar(2,INT32).unwrap(),
    ///     TypedValue::from_scalar(3,INT32).unwrap()];
    /// let tv1 = TypedValue::from_vector(v.clone(), FromVectorMode::Vector).unwrap();
    /// let v2 = tv1.to_vector().unwrap();
    /// assert!(v.iter().zip(v2.iter()).all(|(a,b)| a==b))
    /// ```
    fn to_vector(&self) -> Result<Vec<TypedValue>> {
        let vec_val = self.value.to_vector()?;
        let mut res = vec![];
        match &self.t {
            Type::Tuple(ts) => {
                if ts.len() != vec_val.len() {
                    return Err(runtime_error!(
                        "Inconsistent number of elements: {} vs {}",
                        ts.len(),
                        vec_val.len()
                    ));
                }
                for (t, value) in ts.iter().zip(vec_val.iter()) {
                    res.push(TypedValue::new(t.as_ref().clone(), value.clone())?);
                }
                Ok(res)
            }
            Type::Vector(n, t) => {
                if *n != (vec_val.len() as u64) {
                    return Err(runtime_error!(
                        "Inconsistent number of elements: {} vs {}",
                        *n,
                        vec_val.len()
                    ));
                }
                let mut res = vec![];
                for val in vec_val {
                    res.push(TypedValue::new(t.as_ref().clone(), val)?);
                }
                Ok(res)
            }
            Type::NamedTuple(n_ts) => {
                if n_ts.len() != vec_val.len() {
                    return Err(runtime_error!(
                        "Inconsistent number of elements: {} vs {}",
                        n_ts.len(),
                        vec_val.len()
                    ));
                }
                for (n_t, value) in n_ts.iter().zip(vec_val.iter()) {
                    res.push(TypedValue::new_named(
                        n_t.1.as_ref().clone(),
                        value.clone(),
                        n_t.0.clone(),
                    )?);
                }
                Ok(res)
            }
            _ => Err(runtime_error!("Not a vector: {:?}", self.t)),
        }
    }

    /// Constructs a typed value in ciphercore vector or tuple format from a vector of other typed values.
    ///
    ///
    /// # Arguments
    ///
    /// `v` - vector of typed values
    ///  `mode` - how to decide type of output. if `FromVectorMode::Vector` passed, it constructs a
    ///     ciphercore vector typed value. If `FromVectorMode::Tuple` passed, it constructs a ciphercore
    ///     tuple typed value, And if `FromVectorMode::AutoDetection` passed, it decides based on the type
    ///     of input typed values, if they are all the same type it constructs a vector, otherwise a tuple.
    ///
    /// # Returns
    ///
    /// New typed value constructed from `v`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_types::INT32;
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ciphercore_base::typed_value_operations::{TypedValueOperations,FromVectorMode};
    /// let v = TypedValue::from_vector(
    ///     vec![
    ///         TypedValue::from_scalar(1, INT32).unwrap(),
    ///         TypedValue::from_scalar(423532, INT32).unwrap(),
    ///         TypedValue::from_scalar(-91, INT32).unwrap()],FromVectorMode::Vector).unwrap();
    /// ```
    fn from_vector(v: Vec<TypedValue>, mode: FromVectorMode) -> Result<TypedValue> {
        match mode {
            FromVectorMode::Vector => vector_from_vector_helper(v),
            FromVectorMode::Tuple => tuple_from_vector_helper(v),
            FromVectorMode::AutoDetection => {
                let val_type = v[0].t.clone();
                if v.iter().all(|x| x.t.eq(&val_type)) {
                    vector_from_vector_helper(v)
                } else {
                    tuple_from_vector_helper(v)
                }
            }
        }
    }
    /// If `self` is Vector or Tuple or NamedTuple, gets index_th element of the collection.
    /// Returns error if `self` is Scalar or Array.
    /// Returns error on out of bound access
    ///
    /// # Arguments
    ///
    /// `index` - index of the element in the vector or tuple
    ///
    /// # Returns
    ///
    /// index_th element of the collection
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_types::{INT32,BIT};
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ciphercore_base::typed_value_operations::{TypedValueOperations,FromVectorMode};
    /// let tv = TypedValue::from_vector(
    ///          vec![TypedValue::from_scalar(1, BIT).unwrap(),
    ///          TypedValue::from_scalar(-91, INT32).unwrap()],FromVectorMode::Tuple).unwrap();
    /// let tv1 = tv.get(1).unwrap();
    /// assert!(tv1.eq(&TypedValue::from_scalar(-91, INT32).unwrap()));
    /// ```
    fn get(&self, index: usize) -> Result<TypedValue> {
        get_helper::<TypedValue>(self, index)
    }
    /// If `self` is Vector or Tuple or NamedTuple, insert to_insert_element to index_th position of the collection.
    /// Return error if `self` is Scalar or Array.
    /// Returns error on out of bound access
    /// If inserting to a vector the Type of to_insert_element should match the Type of existing vector's elements
    ///
    /// # Arguments
    ///
    /// `index` - index of the element in the vector or tuple
    /// `to_insert_element` - element to insert
    ///
    /// # Returns
    ///
    /// ()
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_types::{INT32,BIT,UINT64};
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ciphercore_base::typed_value_operations::{TypedValueOperations,FromVectorMode};
    /// let mut tv = TypedValue::from_vector(
    ///          vec![TypedValue::from_scalar(1, BIT).unwrap(),
    ///          TypedValue::from_scalar(-91, INT32).unwrap()], FromVectorMode::Tuple).unwrap();
    /// tv.insert(TypedValue::from_scalar(2,UINT64).unwrap(),1).unwrap();
    /// ```
    fn insert(&mut self, to_insert_element: TypedValue, index: usize) -> Result<()> {
        *self = insert_helper::<TypedValue>(self, to_insert_element, index)?;
        Ok(())
    }
    /// If `self` is Vector or Tuple or NamedTuple, push to_insert_element to the end of the collection.
    /// Return error if `self` is Scalar or Array.
    /// If pushing to a vector, the Type of to_insert_element should match the Type of existing vector's elements
    /// Please note that call to this function copies all the data, as a result pushing N elements
    /// has quadratic complexity.
    ///
    /// # Arguments
    ///
    /// `index` - index of the element in the vector or tuple
    /// `to_insert_element` - element to insert
    ///
    /// # Returns
    ///
    /// ()
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_types::{INT32,BIT,UINT64};
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ciphercore_base::typed_value_operations::{TypedValueOperations,FromVectorMode};
    /// let mut tv = TypedValue::from_vector(
    ///          vec![TypedValue::from_scalar(1, BIT).unwrap(),
    ///          TypedValue::from_scalar(-91, INT32).unwrap()], FromVectorMode::Tuple).unwrap();
    /// tv.push(TypedValue::from_scalar(2,UINT64).unwrap()).unwrap();
    /// ```
    fn push(&mut self, to_insert_element: TypedValue) -> Result<()> {
        *self = push_helper::<TypedValue>(self, to_insert_element)?;
        Ok(())
    }
    /// If `self` is Vector or Tuple or NamedTuple, it extends to_extend_collection to the end of the `self`
    /// Return error if `self` is Scalar or Array.
    /// If extending to a vector the Type of to_extend_collection should match the Type of `self` vector's elements
    ///
    /// # Arguments
    ///
    /// `index` - index of the element in the vector or tuple
    /// `to_extend_collection` - A Vector, Tuple, or NamedTuple to extend to `self`
    ///
    /// # Returns
    ///
    /// ()
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_types::{INT32,BIT,UINT64};
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ciphercore_base::typed_value_operations::{TypedValueOperations,FromVectorMode};
    /// let mut tv = TypedValue::from_vector(
    ///          vec![TypedValue::from_scalar(1, BIT).unwrap(),
    ///          TypedValue::from_scalar(-91, INT32).unwrap()], FromVectorMode::Tuple).unwrap();
    /// let  to_extend_tv = TypedValue::from_vector(
    ///          vec![TypedValue::from_scalar(2, UINT64).unwrap(),
    ///          TypedValue::from_scalar(0, BIT).unwrap()], FromVectorMode::Tuple).unwrap();
    /// tv.extend(to_extend_tv).unwrap();
    /// ```
    fn extend(&mut self, to_extend_collection: TypedValue) -> Result<()> {
        *self = extend_helper::<TypedValue>(self, to_extend_collection)?;
        Ok(())
    }
    /// If `self` is Vector or Tuple or NamedTuple, it removes index_th item of the `self`
    /// Return error if `self` is Scalar or Array.
    /// Returns error on out of bound remove
    ///
    /// # Arguments
    ///
    /// `index` - index of the element in the vector or tuple
    ///
    /// # Returns
    ///
    /// ()
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_types::{INT32,BIT,UINT64};
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ciphercore_base::typed_value_operations::{TypedValueOperations,FromVectorMode};
    /// let mut tv = TypedValue::from_vector(
    ///          vec![TypedValue::from_scalar(1, BIT).unwrap(),
    ///          TypedValue::from_scalar(-91, INT32).unwrap()], FromVectorMode::Tuple).unwrap();
    /// tv.remove(0).unwrap();
    /// ```
    fn remove(&mut self, index: usize) -> Result<()> {
        *self = remove_helper::<TypedValue>(self, index)?;
        Ok(())
    }
    /// If `self` is Vector or Tuple or NamedTuple, it gets then removes index_th item of the `self`
    /// Return error if `self` is Scalar or Array.
    /// Returns error on out of bound remove
    ///
    /// # Arguments
    ///
    /// `index` - index of the element in the vector or tuple
    ///
    /// # Returns
    ///
    /// ()
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_types::{INT32,BIT,UINT64};
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ciphercore_base::typed_value_operations::{TypedValueOperations,FromVectorMode};
    /// let mut tv = TypedValue::from_vector(
    ///          vec![TypedValue::from_scalar(1, BIT).unwrap(),
    ///          TypedValue::from_scalar(-91, INT32).unwrap()], FromVectorMode::Tuple).unwrap();
    /// let tv1 = tv.pop(0).unwrap();
    /// assert!(tv1.eq(&TypedValue::from_scalar(1, BIT).unwrap()));
    /// ```
    fn pop(&mut self, index: usize) -> Result<TypedValue> {
        let (out1, out2) = pop_helper::<TypedValue>(self, index)?;
        *self = out2;
        Ok(out1)
    }
    /// If `self` is Vector or Tuple or NamedTuple, it gets a same collection of typed value with a subset of `self` elements
    /// Return error if `self` is Scalar or Array.
    /// Returns error on out of bound remove
    ///
    /// # Arguments
    ///
    /// `start_index_option` - start index of the element in the vector or tuple, default value is `0` if `None` passed
    /// `start_index_option` - end index of the element in the vector or tuple, default value is end of the vector or tuple if `None` passed
    /// `step_option - step, default value is `1` if `None` passed
    /// # Returns
    ///
    /// A TypedValue which is a sub-vector of TypedValues
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_types::{INT32,BIT,UINT64};
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ciphercore_base::typed_value_operations::{TypedValueOperations,FromVectorMode};
    /// let tv = TypedValue::from_vector(
    ///          vec![TypedValue::from_scalar(1, INT32).unwrap(),
    ///          TypedValue::from_scalar(3, INT32).unwrap(),
    ///          TypedValue::from_scalar(4, INT32).unwrap(),
    ///          TypedValue::from_scalar(5, INT32).unwrap(),
    ///          TypedValue::from_scalar(6, INT32).unwrap(),
    ///          TypedValue::from_scalar(7, INT32).unwrap()],
    ///          FromVectorMode::Vector).unwrap();
    /// let tv1 = tv.get_sub_vector(Some(1),None,Some(2)).unwrap();
    /// ```
    fn get_sub_vector(
        &self,
        start_index_option: Option<usize>,
        end_index_option: Option<usize>,
        step_option: Option<usize>,
    ) -> Result<TypedValue> {
        get_sub_vector_helper::<TypedValue>(self, start_index_option, end_index_option, step_option)
    }
}

impl TypedValueArrayOperations<TypedValue> for TypedValue {
    /// Constructs a typed value from a multi-dimensional bit or integer array.
    ///
    /// # Arguments
    ///
    /// * `x` - array to be converted to a typed value, can have entries of any standard integer type
    /// * `st` - scalar type corresponding to the entries of `x`
    ///
    /// # Returns
    ///
    /// New value constructed from `x`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_types::BIT;
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ndarray::array;
    /// # use ciphercore_base::typed_value_operations::TypedValueArrayOperations;
    /// let a = array![[false, true, true, false], [true, false, false, true]].into_dyn();
    /// let v = TypedValue::from_ndarray(a, BIT).unwrap();
    /// v.value.access_bytes(|bytes| {
    ///     assert_eq!(*bytes, vec![150]);
    ///
    ///     Ok(())
    /// }).unwrap();
    /// ```
    fn from_ndarray<T: TryInto<u128> + Not<Output = T> + TryInto<u8> + Copy>(
        a: ndarray::ArrayD<T>,
        st: ScalarType,
    ) -> Result<Self> {
        let t = array_type(a.shape().iter().map(|x| *x as u64).collect(), st);
        Ok(TypedValue {
            t,
            value: Value::from_ndarray(a, st)?,
            name: None,
        })
    }
}

impl<T> ToNdarray<T> for TypedValue
where
    data_values::Value: data_values::ToNdarray<T>,
{
    fn to_ndarray(&self) -> Result<ndarray::ArrayD<T>> {
        data_values::ToNdarray::<T>::to_ndarray(&self.value, self.t.clone())
    }
}

impl TypedValue {
    /// Creates a typed value from a given type and value.
    /// Checks that the value is a valid for the given type.
    /// Note: check might be not sufficient.
    ///
    /// # Arguments
    ///
    /// `t` - the type
    /// `value` - the value
    ///
    /// # Returns
    ///
    /// New typed value constructed from the given type and value.
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_types::{INT32, Type};
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::typed_value::TypedValue;
    /// let t = Type::Scalar(INT32);
    /// let v = TypedValue::new(t.clone(), Value::zero_of_type(t.clone())).unwrap();
    /// ```
    pub fn new(t: Type, value: Value) -> Result<Self> {
        let type_check = value.check_type(t.clone());
        match type_check {
            Ok(flag) => {
                if !flag {
                    return Err(runtime_error!(
                        "Value doesn't match type. Type is: {t:?}, value is: {value:?}"
                    ));
                }
            }
            Err(_) => {
                return Err(runtime_error!("Cannot check type: {type_check:?}"));
            }
        };
        Ok(TypedValue {
            value,
            t,
            name: None,
        })
    }

    /// Creates a typed value from a given type, value and name.
    /// Checks that the value is a valid for the given type.
    /// Note: check might be not sufficient.
    ///
    /// # Arguments
    ///
    /// `t` - the type
    /// `value` - the value
    /// `name` - the name (can be used to construct named tuple)
    ///
    /// # Returns
    ///
    /// New typed value constructed from the given type,value and name.
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_types::{INT32, Type};
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::typed_value::TypedValue;
    /// let t = Type::Scalar(INT32);
    /// let v = TypedValue::new_named(t.clone(), Value::zero_of_type(t.clone()), "name".to_owned()).unwrap();
    /// ```
    pub fn new_named(t: Type, value: Value, name: String) -> Result<Self> {
        let type_check = value.check_type(t.clone());
        match type_check {
            Ok(flag) => {
                if !flag {
                    return Err(runtime_error!(
                        "Value doesn't match type. Type is: {t:?}, value is: {value:?}"
                    ));
                }
            }
            Err(_) => {
                return Err(runtime_error!("Cannot check type: {type_check:?}"));
            }
        };
        Ok(TypedValue {
            value,
            t,
            name: Some(name),
        })
    }
    /// Generates a typed value of a given type with all-zero bytes.
    ///
    /// # Arguments
    ///
    /// `t` - the type of a new value
    ///
    /// # Returns
    ///
    /// "Zero" typed value of type `t`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_types::{array_type, INT32};
    /// # use ciphercore_base::typed_value::TypedValue;
    /// # use ciphercore_base::typed_value_operations::ToNdarray;
    /// # use ndarray::array;
    /// let v = TypedValue::zero_of_type(array_type(vec![2, 2], INT32));
    /// let a = ToNdarray::<i32>::to_ndarray(&v).unwrap();
    /// assert_eq!(a, array![[0, 0], [0, 0]].into_dyn());
    /// ```
    pub fn zero_of_type(t: Type) -> Self {
        TypedValue {
            value: Value::zero_of_type(t.clone()),
            t,
            name: None,
        }
    }

    /// Constructs a typed value from a given bit or integer scalar.
    ///
    /// # Arguments
    ///
    /// * `x` - scalar to be converted to a value, can be of any standard integer type
    /// * `st` - scalar type corresponding to `x`
    ///
    /// # Returns
    ///
    /// New typed value constructed from `x`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_types::INT32;
    /// # use ciphercore_base::typed_value::TypedValue;
    /// let v = TypedValue::from_scalar(-123456, INT32).unwrap();
    /// v.value.access_bytes(|bytes| {
    ///     assert_eq!(*bytes, vec![192, 29, 254, 255]);
    ///     Ok(())
    /// }).unwrap();
    /// ```
    pub fn from_scalar<T: TryInto<u128> + Not<Output = T> + TryInto<u8> + Copy>(
        x: T,
        st: ScalarType,
    ) -> Result<Self> {
        Ok(TypedValue {
            t: Type::Scalar(st),
            value: Value::from_scalar(x, st)?,
            name: None,
        })
    }

    /// Converts `self` to a scalar if it is a byte vector, then casts the result to `u64`.
    ///
    /// # Arguments
    ///
    /// `st` - scalar type used to interpret `self`
    ///
    /// # Result
    ///
    /// Resulting scalar cast to `u64`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_types::INT32;
    /// # use ciphercore_base::typed_value::TypedValue;
    /// let v = TypedValue::from_scalar(-123456, INT32).unwrap();
    /// assert_eq!(v.to_u64().unwrap(), -123456i32 as u64);
    /// ```
    pub fn to_u64(&self) -> Result<u64> {
        if let Type::Scalar(st) = &self.t {
            Ok(self.value.to_u64(*st)?)
        } else {
            Err(runtime_error!("Cannot convert type {:?} to u64", self.t))
        }
    }

    /// Converts `self` to a scalar if it is a byte vector, then casts the result to `u128`.
    ///
    /// # Arguments
    ///
    /// `st` - scalar type used to interpret `self`
    ///
    /// # Result
    ///
    /// Resulting scalar cast to `u128`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_types::INT32;
    /// # use ciphercore_base::typed_value::TypedValue;
    /// let v = TypedValue::from_scalar(-123456, INT32).unwrap();
    /// assert_eq!(v.to_u128().unwrap(), -123456i32 as u128);
    /// ```
    pub fn to_u128(&self) -> Result<u128> {
        if let Type::Scalar(st) = &self.t {
            Ok(self.value.to_u128(*st)?)
        } else {
            Err(runtime_error!("Cannot convert type {:?} to u128", self.t))
        }
    }

    pub fn secret_share(&self, prng: &mut PRNG) -> Result<TypedValue> {
        let vals = self.shard_to_shares(prng)?;
        Ok(TypedValue {
            t: tuple_type(vec![self.t.clone(), self.t.clone(), self.t.clone()]),
            value: Value::from_vector(vals),
            name: None,
        })
    }

    pub fn get_local_shares_for_each_party(&self, prng: &mut PRNG) -> Result<Vec<TypedValue>> {
        let v = self.shard_to_shares(prng)?;
        let mut garbage = vec![];
        for _ in 0..3 {
            garbage.push(prng.get_random_value(self.t.clone())?);
        }
        Ok(vec![
            TypedValue {
                t: tuple_type(vec![self.t.clone(), self.t.clone(), self.t.clone()]),
                value: Value::from_vector(vec![v[0].clone(), v[1].clone(), garbage[2].clone()]),
                name: None,
            },
            TypedValue {
                t: tuple_type(vec![self.t.clone(), self.t.clone(), self.t.clone()]),
                value: Value::from_vector(vec![garbage[0].clone(), v[1].clone(), v[2].clone()]),
                name: None,
            },
            TypedValue {
                t: tuple_type(vec![self.t.clone(), self.t.clone(), self.t.clone()]),
                value: Value::from_vector(vec![v[0].clone(), garbage[1].clone(), v[2].clone()]),
                name: None,
            },
        ])
    }

    fn shard_to_shares(&self, prng: &mut PRNG) -> Result<Vec<Value>> {
        let v0 = prng.get_random_value(self.t.clone())?;
        let v1 = prng.get_random_value(self.t.clone())?;
        let v2 = generalized_subtract(
            generalized_subtract(self.value.clone(), v0.clone(), self.t.clone())?,
            v1.clone(),
            self.t.clone(),
        )?;
        Ok(vec![v0, v1, v2])
    }

    pub fn secret_share_reveal(&self) -> Result<TypedValue> {
        if let Type::Tuple(tv) = self.t.clone() {
            if tv.len() == 3 && tv[0] == tv[1] && tv[0] == tv[2] {
                let (v0, v1, v2) = self
                    .value
                    .access_vector(|vv| Ok((vv[0].clone(), vv[1].clone(), vv[2].clone())))?;
                let v_output = generalized_add(
                    generalized_add(v0, v1, (*tv[0]).clone())?,
                    v2,
                    (*tv[0]).clone(),
                )?;
                Ok(TypedValue::new((*tv[0]).clone(), v_output)?)
            } else {
                Err(runtime_error!("Not a secret-shared type/value"))
            }
        } else {
            Err(runtime_error!("Not a secret-shared type/value"))
        }
    }
}

pub fn generalized_subtract(v: Value, v0: Value, t: Type) -> Result<Value> {
    match t {
        Type::Scalar(st) | Type::Array(_, st) => {
            let v_raw = v.access_bytes(|bytes| vec_u128_from_bytes(bytes, st))?;
            let v0_raw = v0.access_bytes(|bytes| vec_u128_from_bytes(bytes, st))?;
            let result = subtract_vectors_u128(&v_raw, &v0_raw, st.get_modulus())?;
            Ok(Value::from_bytes(vec_to_bytes(&result, st)?))
        }
        Type::Tuple(tv) => {
            let v_raw = v.access_vector(|vector| Ok(vector.clone()))?;
            let v0_raw = v0.access_vector(|vector| Ok(vector.clone()))?;
            let mut result = vec![];
            for i in 0..tv.len() {
                result.push(generalized_subtract(
                    v_raw[i].clone(),
                    v0_raw[i].clone(),
                    (*tv[i]).clone(),
                )?);
            }
            Ok(Value::from_vector(result))
        }
        Type::NamedTuple(tv) => {
            let v_raw = v.access_vector(|vector| Ok(vector.clone()))?;
            let v0_raw = v0.access_vector(|vector| Ok(vector.clone()))?;
            let mut result = vec![];
            for i in 0..tv.len() {
                result.push(generalized_subtract(
                    v_raw[i].clone(),
                    v0_raw[i].clone(),
                    (*tv[i].1).clone(),
                )?);
            }
            Ok(Value::from_vector(result))
        }
        Type::Vector(len, element_type) => {
            let v_raw = v.access_vector(|vector| Ok(vector.clone()))?;
            let v0_raw = v0.access_vector(|vector| Ok(vector.clone()))?;
            let mut result = vec![];
            for i in 0..(len as usize) {
                result.push(generalized_subtract(
                    v_raw[i].clone(),
                    v0_raw[i].clone(),
                    (*element_type).clone(),
                )?);
            }
            Ok(Value::from_vector(result))
        }
    }
}

pub fn generalized_add(v: Value, v0: Value, t: Type) -> Result<Value> {
    match t {
        Type::Scalar(st) | Type::Array(_, st) => {
            let v_raw = v.access_bytes(|bytes| vec_u128_from_bytes(bytes, st))?;
            let v0_raw = v0.access_bytes(|bytes| vec_u128_from_bytes(bytes, st))?;
            let result = add_vectors_u128(&v_raw, &v0_raw, st.get_modulus())?;
            Ok(Value::from_bytes(vec_to_bytes(&result, st)?))
        }
        Type::Tuple(tv) => {
            let v_raw = v.access_vector(|vector| Ok(vector.clone()))?;
            let v0_raw = v0.access_vector(|vector| Ok(vector.clone()))?;
            let mut result = vec![];
            for i in 0..tv.len() {
                result.push(generalized_add(
                    v_raw[i].clone(),
                    v0_raw[i].clone(),
                    (*tv[i]).clone(),
                )?);
            }
            Ok(Value::from_vector(result))
        }
        Type::NamedTuple(tv) => {
            let v_raw = v.access_vector(|vector| Ok(vector.clone()))?;
            let v0_raw = v0.access_vector(|vector| Ok(vector.clone()))?;
            let mut result = vec![];
            for i in 0..tv.len() {
                result.push(generalized_add(
                    v_raw[i].clone(),
                    v0_raw[i].clone(),
                    (*tv[i].1).clone(),
                )?);
            }
            Ok(Value::from_vector(result))
        }
        Type::Vector(len, element_type) => {
            let v_raw = v.access_vector(|vector| Ok(vector.clone()))?;
            let v0_raw = v0.access_vector(|vector| Ok(vector.clone()))?;
            let mut result = vec![];
            for i in 0..(len as usize) {
                result.push(generalized_add(
                    v_raw[i].clone(),
                    v0_raw[i].clone(),
                    (*element_type).clone(),
                )?);
            }
            Ok(Value::from_vector(result))
        }
    }
}

fn vector_from_vector_helper(v: Vec<TypedValue>) -> Result<TypedValue> {
    let val_type = if v.is_empty() {
        tuple_type(vec![])
    } else {
        v[0].t.clone()
    };

    let t = vector_type(v.len() as u64, val_type.clone());
    let mut val_vec = vec![];
    for val in v {
        if !val.t.eq(&val_type) {
            return Err(runtime_error!(
                "Can not distinguish the type: vector has different types"
            ));
        }
        val_vec.push(val.value);
    }
    Ok(TypedValue {
        t,
        value: Value::from_vector(val_vec),
        name: None,
    })
}
fn tuple_from_vector_helper(v: Vec<TypedValue>) -> Result<TypedValue> {
    let named = v.iter().all(|x| x.name.is_some());
    let unnamed = v.iter().all(|x| x.name.is_none());
    if unnamed {
        let t = tuple_type(v.iter().map(|v| v.t.clone()).collect());
        let val_vec = v.iter().map(|v| v.value.clone()).collect();
        Ok(TypedValue::new(t, Value::from_vector(val_vec))?)
    } else if named {
        let t = named_tuple_type(
            v.iter()
                .map(|v| (v.name.clone().unwrap(), v.t.clone()))
                .collect(),
        );
        let val_vec = v.iter().map(|v| v.value.clone()).collect();
        Ok(TypedValue::new(t, Value::from_vector(val_vec))?)
    } else {
        Err(runtime_error!("Can not mix named and unnamed tuple"))
    }
}
#[cfg(test)]
mod tests {
    use std::ops::Add;

    use super::*;
    use crate::data_types::{
        scalar_type, tuple_type, BIT, INT32, INT8, UINT128, UINT16, UINT32, UINT64, UINT8,
    };

    #[test]
    fn test_secret_sharing() {
        || -> Result<()> {
            let mut prng = PRNG::new(None)?;
            for bv in 0..2 {
                let tv = TypedValue::new(scalar_type(BIT), Value::from_bytes(vec![bv]))?;
                for _ in 0..10 {
                    let tv1 = tv.secret_share(&mut prng)?;
                    let mut result = vec![];
                    tv1.value.access_vector(|v| {
                        assert_eq!(v.len(), 3);
                        for i in 0..3 {
                            v[i].access_bytes(|b| {
                                assert_eq!(b.len(), 1);
                                result.push(b[0]);
                                Ok(())
                            })?;
                        }
                        Ok(())
                    })?;
                    assert_eq!(result[0] ^ result[1] ^ result[2], bv);
                    assert_eq!(tv1.secret_share_reveal()?, tv);
                }
            }
            for _ in 0..10 {
                let tv = TypedValue::new(
                    scalar_type(INT32),
                    prng.get_random_value(scalar_type(INT32))?,
                )?;
                for _ in 0..10 {
                    let tv1 = tv.secret_share(&mut prng)?;
                    let mut result = vec![];
                    tv1.value.access_vector(|v| {
                        assert_eq!(v.len(), 3);
                        result.push(v[0].to_i32(INT32)?);
                        result.push(v[1].to_i32(INT32)?);
                        result.push(v[2].to_i32(INT32)?);
                        Ok(())
                    })?;
                    assert_eq!(
                        result[0]
                            .overflowing_add(result[1])
                            .0
                            .overflowing_add(result[2])
                            .0,
                        tv.value.to_i32(INT32)?
                    );
                    assert_eq!(tv1.secret_share_reveal()?, tv);
                }
            }
            for _ in 0..10 {
                let t = tuple_type(vec![scalar_type(INT32), scalar_type(BIT)]);
                let tv = TypedValue::new(t.clone(), prng.get_random_value(t.clone())?)?;
                for _ in 0..10 {
                    let tv1 = tv.secret_share(&mut prng)?;
                    let mut result = vec![];
                    tv1.value.access_vector(|v| {
                        assert_eq!(v.len(), 3);
                        for i in 0..3 {
                            v[i].access_vector(|v1| {
                                assert_eq!(v1.len(), 2);
                                result.push((v1[0].to_i32(INT32)?, v1[1].to_u8(BIT)?));
                                Ok(())
                            })?;
                        }
                        Ok(())
                    })?;
                    let ov = tv
                        .value
                        .access_vector(|v| Ok((v[0].to_i32(INT32)?, v[1].to_u8(BIT)?)))?;
                    assert_eq!(
                        result[0]
                            .0
                            .overflowing_add(result[1].0)
                            .0
                            .overflowing_add(result[2].0)
                            .0,
                        ov.0
                    );
                    assert_eq!(result[0].1 ^ result[1].1 ^ result[2].1, ov.1);
                    assert_eq!(tv1.secret_share_reveal()?, tv);
                }
            }
            for _ in 0..10 {
                let t = named_tuple_type(vec![
                    ("field1".to_owned(), scalar_type(INT32)),
                    ("field2".to_owned(), scalar_type(BIT)),
                ]);
                let tv = TypedValue::new(t.clone(), prng.get_random_value(t.clone())?)?;
                for _ in 0..10 {
                    let tv1 = tv.secret_share(&mut prng)?;
                    let mut result = vec![];
                    tv1.value.access_vector(|v| {
                        assert_eq!(v.len(), 3);
                        for i in 0..3 {
                            v[i].access_vector(|v1| {
                                assert_eq!(v1.len(), 2);
                                result.push((v1[0].to_i32(INT32)?, v1[1].to_u8(BIT)?));
                                Ok(())
                            })?;
                        }
                        Ok(())
                    })?;
                    let ov = tv
                        .value
                        .access_vector(|v| Ok((v[0].to_i32(INT32)?, v[1].to_u8(BIT)?)))?;
                    assert_eq!(
                        result[0]
                            .0
                            .overflowing_add(result[1].0)
                            .0
                            .overflowing_add(result[2].0)
                            .0,
                        ov.0
                    );
                    assert_eq!(result[0].1 ^ result[1].1 ^ result[2].1, ov.1);
                    assert_eq!(tv1.secret_share_reveal()?, tv);
                }
            }
            for _ in 0..10 {
                let t = vector_type(2, scalar_type(INT32));
                let tv = TypedValue::new(t.clone(), prng.get_random_value(t.clone())?)?;
                for _ in 0..10 {
                    let tv1 = tv.secret_share(&mut prng)?;
                    let mut result = vec![];
                    tv1.value.access_vector(|v| {
                        assert_eq!(v.len(), 3);
                        for i in 0..3 {
                            v[i].access_vector(|v1| {
                                assert_eq!(v1.len(), 2);
                                result.push((v1[0].to_i32(INT32)?, v1[1].to_i32(INT32)?));
                                Ok(())
                            })?;
                        }
                        Ok(())
                    })?;
                    let ov = tv
                        .value
                        .access_vector(|v| Ok((v[0].to_i32(INT32)?, v[1].to_i32(INT32)?)))?;
                    assert_eq!(
                        result[0]
                            .0
                            .overflowing_add(result[1].0)
                            .0
                            .overflowing_add(result[2].0)
                            .0,
                        ov.0
                    );
                    assert_eq!(
                        result[0]
                            .1
                            .overflowing_add(result[1].1)
                            .0
                            .overflowing_add(result[2].1)
                            .0,
                        ov.1
                    );
                    assert_eq!(tv1.secret_share_reveal()?, tv);
                }
            }
            for _ in 0..10 {
                let t = array_type(vec![2], INT32);
                let tv = TypedValue::new(t.clone(), prng.get_random_value(t.clone())?)?;
                for _ in 0..10 {
                    let tv1 = tv.secret_share(&mut prng)?;
                    let mut result = vec![];
                    tv1.value.access_vector(|v| {
                        assert_eq!(v.len(), 3);
                        for i in 0..3 {
                            result.push(v[i].to_flattened_array_i32(t.clone())?);
                        }
                        Ok(())
                    })?;
                    let ov = tv.value.to_flattened_array_i32(t.clone())?;
                    assert_eq!(
                        result[0][0]
                            .overflowing_add(result[1][0])
                            .0
                            .overflowing_add(result[2][0])
                            .0,
                        ov[0]
                    );
                    assert_eq!(
                        result[0][1]
                            .overflowing_add(result[1][1])
                            .0
                            .overflowing_add(result[2][1])
                            .0,
                        ov[1]
                    );
                    assert_eq!(tv1.secret_share_reveal()?, tv);
                }
            }
            assert!(
                TypedValue::new(scalar_type(BIT), Value::from_scalar(0, BIT)?)?
                    .secret_share_reveal()
                    .is_err()
            );
            assert!(
                TypedValue::new(tuple_type(vec![]), Value::from_vector(vec![]))?
                    .secret_share_reveal()
                    .is_err()
            );
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_get_secret_shares() {
        || -> Result<()> {
            let mut prng = PRNG::new(None)?;
            let data = vec![12, 34, 56];
            let value = TypedValue::from_ndarray(
                ndarray::Array::from_vec(data.clone()).into_dyn(),
                UINT64,
            )?;
            let shares = value.get_local_shares_for_each_party(&mut prng)?;
            let shares0 = shares[0].to_vector()?;
            let shares1 = shares[1].to_vector()?;
            let shares2 = shares[2].to_vector()?;
            assert_eq!(shares0[0], shares2[0]);
            assert_eq!(shares0[1], shares1[1]);
            assert_eq!(shares1[2], shares2[2]);
            let m = UINT32.get_modulus().unwrap();
            let v0 = ToNdarray::<u128>::to_ndarray(&shares0[0])? % m;
            let v1 = ToNdarray::<u128>::to_ndarray(&shares1[1])? % m;
            let v2 = ToNdarray::<u128>::to_ndarray(&shares2[2])? % m;
            let new_data = v0.add(v1).add(v2) % m;
            assert_eq!(new_data, ndarray::Array::from_vec(data).into_dyn());
            Ok(())
        }()
        .unwrap();
    }
    #[test]
    fn test_is_equal() {
        // two identical integers are equal
        || -> Result<()> {
            let t1 = scalar_type(INT8);
            let t2 = scalar_type(INT8);
            let v1 = Value::from_bytes(vec![13]);
            let v2 = Value::from_bytes(vec![13]);
            let tv1 = TypedValue::new(t1, v1)?;
            let tv2 = TypedValue::new(t2, v2)?;
            assert!(tv1.is_equal(&tv2)?);
            Ok(())
        }()
        .unwrap();
        // two distinct integers are not equal
        || -> Result<()> {
            let t1 = scalar_type(INT8);
            let t2 = scalar_type(INT8);
            let v1 = Value::from_bytes(vec![11]);
            let v2 = Value::from_bytes(vec![13]);
            let tv1 = TypedValue::new(t1, v1)?;
            let tv2 = TypedValue::new(t2, v2)?;
            assert!(!tv1.is_equal(&tv2)?);
            Ok(())
        }()
        .unwrap();
        // integers with different types are not equal
        || -> Result<()> {
            let t1 = scalar_type(INT8);
            let t2 = scalar_type(UINT8);
            let v1 = Value::from_bytes(vec![13]);
            let v2 = Value::from_bytes(vec![13]);
            let tv1 = TypedValue::new(t1, v1)?;
            let tv2 = TypedValue::new(t2, v2)?;
            assert!(!tv1.is_equal(&tv2)?);
            Ok(())
        }()
        .unwrap();
        // identical arrays are equal
        || -> Result<()> {
            let t1 = array_type(vec![2, 2], BIT);
            let t2 = array_type(vec![2, 2], BIT);
            let v1 = Value::from_bytes(vec![13]);
            let v2 = Value::from_bytes(vec![13]);
            let tv1 = TypedValue::new(t1, v1)?;
            let tv2 = TypedValue::new(t2, v2)?;
            assert!(tv1.is_equal(&tv2)?);
            Ok(())
        }()
        .unwrap();
        // distinct arrays are equal if they are equal up to number of bits in their size
        || -> Result<()> {
            let t1 = array_type(vec![2, 2], BIT);
            let t2 = array_type(vec![2, 2], BIT);
            let v1 = Value::from_bytes(vec![13]);
            let v2 = Value::from_bytes(vec![173]);
            let tv1 = TypedValue::new(t1, v1)?;
            let tv2 = TypedValue::new(t2, v2)?;
            assert!(tv1.is_equal(&tv2)?);
            Ok(())
        }()
        .unwrap();
        // distinct arrays are not equal if they are not equal up to number of bits in their size
        || -> Result<()> {
            let t1 = array_type(vec![2, 2], BIT);
            let t2 = array_type(vec![2, 2], BIT);
            let v1 = Value::from_bytes(vec![13]);
            let v2 = Value::from_bytes(vec![174]);
            let tv1 = TypedValue::new(t1, v1)?;
            let tv2 = TypedValue::new(t2, v2)?;
            assert!(!tv1.is_equal(&tv2)?);
            Ok(())
        }()
        .unwrap();
        // Identical tuples are equal
        || -> Result<()> {
            let t1 = tuple_type(vec![scalar_type(BIT), scalar_type(INT32)]);
            let t2 = t1.clone();
            let v1 = Value::from_vector(vec![
                Value::from_scalar(0, BIT)?,
                Value::from_scalar(-73, INT32)?,
            ]);
            let v2 = Value::from_vector(vec![
                Value::from_scalar(0, BIT)?,
                Value::from_scalar(-73, INT32)?,
            ]);
            let tv1 = TypedValue::new(t1, v1)?;
            let tv2 = TypedValue::new(t2, v2)?;
            assert!(tv1.is_equal(&tv2)?);
            Ok(())
        }()
        .unwrap();
        // Identical tuples are equal, elements are compared with type size awareness
        || -> Result<()> {
            let t1 = tuple_type(vec![scalar_type(BIT), array_type(vec![5, 2], BIT)]);
            let t2 = t1.clone();
            let v1 = Value::from_vector(vec![
                Value::from_scalar(0, BIT)?,
                Value::from_bytes(vec![234, 85]),
            ]);
            let v2 = Value::from_vector(vec![
                Value::from_scalar(0, BIT)?,
                Value::from_bytes(vec![234, 17]),
            ]);
            let tv1 = TypedValue::new(t1, v1)?;
            let tv2 = TypedValue::new(t2, v2)?;
            assert!(tv1.is_equal(&tv2)?);
            Ok(())
        }()
        .unwrap();
        || -> Result<()> {
            let t1 = tuple_type(vec![scalar_type(BIT), array_type(vec![5, 2], BIT)]);
            let t2 = t1.clone();
            let v1 = Value::from_vector(vec![
                Value::from_scalar(0, BIT)?,
                Value::from_bytes(vec![234, 85]),
            ]);
            let v2 = Value::from_vector(vec![
                Value::from_scalar(0, BIT)?,
                Value::from_bytes(vec![234, 14]),
            ]);
            let tv1 = TypedValue::new(t1, v1)?;
            let tv2 = TypedValue::new(t2, v2)?;
            assert!(!tv1.is_equal(&tv2)?);
            Ok(())
        }()
        .unwrap();
    }
    #[test]
    fn test_new() {
        let t0 = Type::Scalar(INT32);
        let t1 = Type::Scalar(UINT32);
        let t2 = Type::Scalar(INT8);
        // Ok cases:
        assert!(TypedValue::new(t0.clone(), Value::zero_of_type(t0.clone())).is_ok());
        assert!(TypedValue::new(t0.clone(), Value::zero_of_type(t1.clone())).is_ok());
        assert!(TypedValue::new(t1, Value::zero_of_type(t0.clone())).is_ok());
        // Err cases:
        assert!(TypedValue::new(t0.clone(), Value::zero_of_type(t2.clone())).is_err());
        assert!(TypedValue::new(t2, Value::zero_of_type(t0)).is_err());
    }
    #[test]
    fn test_zero_of_type() {
        let types = vec![
            scalar_type(BIT),
            scalar_type(INT32),
            scalar_type(UINT64),
            array_type(vec![2, 3], BIT),
            array_type(vec![1, 7], INT32),
            array_type(vec![10, 10], UINT64),
            tuple_type(vec![]),
            tuple_type(vec![scalar_type(BIT), scalar_type(BIT)]),
            tuple_type(vec![scalar_type(INT32), scalar_type(BIT)]),
            tuple_type(vec![tuple_type(vec![]), array_type(vec![5, 5], INT32)]),
            vector_type(10, tuple_type(vec![])),
            vector_type(
                10,
                tuple_type(vec![vector_type(5, scalar_type(INT32)), scalar_type(BIT)]),
            ),
            named_tuple_type(vec![
                ("field 1".to_string(), scalar_type(BIT)),
                ("field 2".to_string(), scalar_type(INT32)),
            ]),
        ];
        for t in types {
            let v = TypedValue::zero_of_type(t.clone());
            assert_eq!(t, v.t);
            assert!(v.value.check_type(v.t.clone()).unwrap());
        }
    }
    #[test]
    fn test_extract_scalar_bit() {
        let v = TypedValue::from_scalar(1, BIT).unwrap();
        let result = v.to_u64().unwrap();
        assert_eq!(result, 1);
    }
    #[test]
    fn test_create_from_scalar() {
        assert_eq!(
            TypedValue::from_scalar(73, UINT64)
                .unwrap()
                .to_u64()
                .unwrap(),
            73
        );
        assert_eq!(
            TypedValue::from_scalar(73, UINT32)
                .unwrap()
                .to_u64()
                .unwrap(),
            73
        );
        assert_eq!(
            TypedValue::from_scalar(-73, INT32)
                .unwrap()
                .to_u64()
                .unwrap(),
            -73i32 as u64
        );
        let x = 131487715213352547813154;
        assert_eq!(
            TypedValue::from_scalar(x, UINT128)
                .unwrap()
                .to_u128()
                .unwrap(),
            x
        );
    }
    #[test]
    fn test_create_from_vector() {
        assert!(TypedValue::from_vector(
            vec![
                TypedValue::from_scalar(0, BIT).unwrap(),
                TypedValue::from_scalar(73, INT32).unwrap(),
            ],
            FromVectorMode::Vector
        )
        .is_err());

        let v = TypedValue::from_vector(
            vec![
                TypedValue::from_scalar(0, INT32).unwrap(),
                TypedValue::from_scalar(73, INT32).unwrap(),
            ],
            FromVectorMode::Vector,
        )
        .unwrap();
        let v2 = TypedValue::from_vector(
            vec![
                TypedValue::from_scalar(0, INT32).unwrap(),
                TypedValue::from_scalar(73, INT32).unwrap(),
            ],
            FromVectorMode::AutoDetection,
        )
        .unwrap();
        assert!(v2.is_equal(&v).unwrap());
        let entries = v.to_vector().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].to_u64().unwrap(), 0);
        assert_eq!(entries[1].to_u64().unwrap(), 73);
        let v3 = TypedValue::from_vector(
            vec![
                TypedValue::from_scalar(0, BIT).unwrap(),
                TypedValue::from_scalar(73, INT32).unwrap(),
            ],
            FromVectorMode::AutoDetection,
        )
        .unwrap();
        assert!(v3.t.is_tuple());
    }
    #[test]
    fn test_ndarray() {
        || -> Result<()> {
            let a = ndarray::Array::from_shape_vec((2, 3), vec![10, -20, 30, 40, -50, 60])?;
            let v = TypedValue::from_ndarray(a.into_dyn(), INT32)?;
            let b = ToNdarray::<i32>::to_ndarray(&v)?;
            assert_eq!(b.shape(), vec![2, 3]);
            assert_eq!(b.into_raw_vec(), vec![10, -20, 30, 40, -50, 60]);
            Ok(())
        }()
        .unwrap();
    }
    #[test]
    fn test_get() {
        //test get from tuple success/failure
        || -> Result<()> {
            let t0 = tuple_type(vec![scalar_type(UINT8), scalar_type(UINT16)]);
            let zero0 = TypedValue::zero_of_type(t0);
            let res_ok = zero0.get(1);
            //ok case
            assert!(res_ok.is_ok());
            let tv_expected = TypedValue::from_scalar(0, UINT16)?;
            assert!(tv_expected.is_equal(&res_ok?)?);
            // out of bound access case
            let res_err = zero0.get(4);
            assert!(res_err.is_err());
            Ok(())
        }()
        .unwrap();
        //test get from vector success/failure
        || -> Result<()> {
            let t0 = vector_type(2, scalar_type(UINT8));
            let zero0 = TypedValue::zero_of_type(t0);
            let res_ok = zero0.get(1);
            //ok case
            assert!(res_ok.is_ok());
            let tv_expected = TypedValue::from_scalar(0, UINT8)?;
            assert!(tv_expected.is_equal(&res_ok?)?);
            // out of bound access case
            let res_err = zero0.get(4);
            assert!(res_err.is_err());
            Ok(())
        }()
        .unwrap();
    }
    #[test]
    fn test_insert() {
        //test insert to tuple success/failure
        || -> Result<()> {
            let t0 = tuple_type(vec![scalar_type(UINT8), scalar_type(UINT16)]);
            let mut zero0 = TypedValue::zero_of_type(t0);
            let to_insert = TypedValue::from_scalar(20, UINT32)?;
            let res_ok = zero0.insert(to_insert.clone(), 1);
            //ok case
            assert!(res_ok.is_ok());
            let t_expected = tuple_type(vec![
                scalar_type(UINT8),
                scalar_type(UINT32),
                scalar_type(UINT16),
            ]);
            let value_expected = Value::from_vector(vec![
                Value::from_scalar(0, UINT8).unwrap(),
                Value::from_scalar(20, UINT32).unwrap(),
                Value::from_scalar(0, UINT16).unwrap(),
            ]);
            let tv_expected = TypedValue::new(t_expected, value_expected)?;
            assert!(tv_expected.is_equal(&zero0)?);
            // out of bound insert case
            let res_err = zero0.insert(to_insert, 4);
            assert!(res_err.is_err());
            Ok(())
        }()
        .unwrap();
        //test insert to vector success/failure
        || -> Result<()> {
            let t0 = vector_type(2, scalar_type(UINT8));
            let mut zero0 = TypedValue::zero_of_type(t0);
            let to_insert = TypedValue::from_scalar(20, UINT8)?;
            let res_ok = zero0.insert(to_insert.clone(), 1);
            //ok case
            assert!(res_ok.is_ok());
            let t_expected = vector_type(3, scalar_type(UINT8));
            let value_expected = Value::from_vector(vec![
                Value::from_scalar(0, UINT8).unwrap(),
                Value::from_scalar(20, UINT8).unwrap(),
                Value::from_scalar(0, UINT8).unwrap(),
            ]);
            let tv_expected = TypedValue::new(t_expected, value_expected)?;
            assert!(tv_expected.is_equal(&zero0)?);
            // out of bound insert case
            let res_err = zero0.insert(to_insert, 4);
            assert!(res_err.is_err());
            // wrong type insert case
            let to_insert = TypedValue::from_scalar(20, UINT16)?;
            let res_err = zero0.insert(to_insert, 1);
            assert!(res_err.is_err());

            Ok(())
        }()
        .unwrap();
        //test insert to incompatible types
        || -> Result<()> {
            //insert to scalar
            let t0 = scalar_type(UINT8);
            let mut zero0 = TypedValue::zero_of_type(t0);
            let to_insert = TypedValue::from_scalar(20, UINT8)?;
            let res_err = zero0.insert(to_insert, 1);
            assert!(res_err.is_err());
            //insert to array
            let t0 = array_type(vec![2, 2], UINT8);
            let mut zero0 = TypedValue::zero_of_type(t0);
            let to_insert = TypedValue::from_scalar(20, UINT8)?;
            let res_err = zero0.insert(to_insert, 1);
            assert!(res_err.is_err());
            Ok(())
        }()
        .unwrap();
    }
    #[test]
    fn test_push() {
        //test push to tuple success
        || -> Result<()> {
            let t0 = tuple_type(vec![scalar_type(UINT8), scalar_type(UINT16)]);
            let mut zero0 = TypedValue::zero_of_type(t0);
            let to_insert = TypedValue::from_scalar(20, UINT32)?;
            let res_ok = zero0.push(to_insert);
            //ok case
            assert!(res_ok.is_ok());
            let tv_expected = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(0, UINT8).unwrap(),
                    TypedValue::from_scalar(0, UINT16).unwrap(),
                    TypedValue::from_scalar(20, UINT32).unwrap(),
                ],
                FromVectorMode::Tuple,
            )
            .unwrap();
            assert!(tv_expected.is_equal(&zero0)?);
            Ok(())
        }()
        .unwrap();
        //test push to tuple fail
        || -> Result<()> {
            let t0 = tuple_type(vec![scalar_type(UINT8), scalar_type(UINT16)]);
            let mut zero0 = TypedValue::zero_of_type(t0);
            let mut to_insert = TypedValue::from_scalar(20, UINT32)?;
            to_insert.name = Some("name".to_owned());
            let res_ok = zero0.push(to_insert);
            //err case
            assert!(res_ok.is_err());
            Ok(())
        }()
        .unwrap();
        //test push to named tuple success
        || -> Result<()> {
            let t0 = named_tuple_type(vec![
                ("name1".to_owned(), scalar_type(UINT8)),
                ("name2".to_owned(), scalar_type(UINT16)),
            ]);
            let mut zero0 = TypedValue::zero_of_type(t0);
            let mut to_insert = TypedValue::from_scalar(20, UINT32)?;
            to_insert.name = Some("name".to_owned());
            let res_ok = zero0.push(to_insert.clone());
            //ok case
            assert!(res_ok.is_ok());
            let mut tv1 = TypedValue::from_scalar(0, UINT8).unwrap();
            tv1.name = Some("name1".to_owned());
            let mut tv2 = TypedValue::from_scalar(0, UINT16).unwrap();
            tv2.name = Some("name2".to_owned());
            let tv3 = to_insert;
            let tv_expected =
                TypedValue::from_vector(vec![tv1, tv2, tv3], FromVectorMode::Tuple).unwrap();
            assert!(tv_expected.is_equal(&zero0)?);
            Ok(())
        }()
        .unwrap();
    }
    #[test]
    fn test_remove() {
        //test remove from tuple
        || -> Result<()> {
            let mut tv = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(0, UINT8).unwrap(),
                    TypedValue::from_scalar(0, UINT16).unwrap(),
                    TypedValue::from_scalar(20, UINT32).unwrap(),
                ],
                FromVectorMode::Tuple,
            )
            .unwrap();
            tv.remove(2)?;
            let tv_expected = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(0, UINT8).unwrap(),
                    TypedValue::from_scalar(0, UINT16).unwrap(),
                ],
                FromVectorMode::Tuple,
            )
            .unwrap();
            assert!(tv_expected.is_equal(&tv)?);
            //failure case : out of bound
            assert!(tv.remove(2).is_err());
            Ok(())
        }()
        .unwrap();
    }
    #[test]
    fn test_pop() {
        //test pop from vector
        || -> Result<()> {
            let mut tv = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(0, UINT8).unwrap(),
                    TypedValue::from_scalar(0, UINT8).unwrap(),
                    TypedValue::from_scalar(20, UINT8).unwrap(),
                ],
                FromVectorMode::Vector,
            )
            .unwrap();
            let element = tv.pop(2)?;
            let tv_expected = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(0, UINT8).unwrap(),
                    TypedValue::from_scalar(0, UINT8).unwrap(),
                ],
                FromVectorMode::Vector,
            )
            .unwrap();
            assert!(tv_expected.is_equal(&tv)?);
            assert!(element.is_equal(&TypedValue::from_scalar(20, UINT8).unwrap())?);
            //failure case : out of bound
            assert!(tv.pop(2).is_err());
            Ok(())
        }()
        .unwrap();
    }
    #[test]
    fn test_extend() {
        //test extend vector to vector
        || -> Result<()> {
            let mut tv = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(0, UINT8).unwrap(),
                    TypedValue::from_scalar(10, UINT8).unwrap(),
                    TypedValue::from_scalar(20, UINT8).unwrap(),
                ],
                FromVectorMode::Vector,
            )
            .unwrap();
            let tv_to_extend = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(30, UINT8).unwrap(),
                    TypedValue::from_scalar(40, UINT8).unwrap(),
                ],
                FromVectorMode::Vector,
            )
            .unwrap();
            let tv_expected = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(0, UINT8).unwrap(),
                    TypedValue::from_scalar(10, UINT8).unwrap(),
                    TypedValue::from_scalar(20, UINT8).unwrap(),
                    TypedValue::from_scalar(30, UINT8).unwrap(),
                    TypedValue::from_scalar(40, UINT8).unwrap(),
                ],
                FromVectorMode::Vector,
            )
            .unwrap();
            tv.extend(tv_to_extend)?;
            assert!(tv_expected.is_equal(&tv)?);
            Ok(())
        }()
        .unwrap();
        //test extend incompatible vector to vector: failure
        || -> Result<()> {
            let mut tv = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(0, UINT8).unwrap(),
                    TypedValue::from_scalar(10, UINT8).unwrap(),
                    TypedValue::from_scalar(20, UINT8).unwrap(),
                ],
                FromVectorMode::Vector,
            )
            .unwrap();
            let tv_to_extend = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(30, UINT16).unwrap(),
                    TypedValue::from_scalar(40, UINT16).unwrap(),
                ],
                FromVectorMode::Vector,
            )
            .unwrap();
            let res = tv.extend(tv_to_extend);
            assert!(res.is_err());
            Ok(())
        }()
        .unwrap();
        //test extend incompatible vector to tuple: success
        || -> Result<()> {
            let mut tv = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(0, UINT8).unwrap(),
                    TypedValue::from_scalar(10, UINT8).unwrap(),
                    TypedValue::from_scalar(20, UINT8).unwrap(),
                ],
                FromVectorMode::Tuple,
            )
            .unwrap();
            let tv_to_extend = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(30, UINT16).unwrap(),
                    TypedValue::from_scalar(40, UINT16).unwrap(),
                ],
                FromVectorMode::Vector,
            )
            .unwrap();
            let res = tv.extend(tv_to_extend);
            assert!(res.is_ok());
            let tv_expected = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(0, UINT8).unwrap(),
                    TypedValue::from_scalar(10, UINT8).unwrap(),
                    TypedValue::from_scalar(20, UINT8).unwrap(),
                    TypedValue::from_scalar(30, UINT16).unwrap(),
                    TypedValue::from_scalar(40, UINT16).unwrap(),
                ],
                FromVectorMode::Tuple,
            )
            .unwrap();
            assert!(tv.is_equal(&tv_expected)?);
            Ok(())
        }()
        .unwrap();
    }
    #[test]
    fn test_get_sub_vector() {
        || -> Result<()> {
            let tv = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(0, UINT16).unwrap(),
                    TypedValue::from_scalar(1, UINT16).unwrap(),
                    TypedValue::from_scalar(2, UINT16).unwrap(),
                    TypedValue::from_scalar(3, UINT16).unwrap(),
                    TypedValue::from_scalar(4, UINT16).unwrap(),
                    TypedValue::from_scalar(5, UINT16).unwrap(),
                    TypedValue::from_scalar(6, UINT16).unwrap(),
                    TypedValue::from_scalar(7, UINT16).unwrap(),
                ],
                FromVectorMode::Vector,
            )
            .unwrap();
            let sub_v = tv.get_sub_vector(None, None, Some(2))?;

            let tv_expected = TypedValue::from_vector(
                vec![
                    TypedValue::from_scalar(0, UINT16).unwrap(),
                    TypedValue::from_scalar(2, UINT16).unwrap(),
                    TypedValue::from_scalar(4, UINT16).unwrap(),
                    TypedValue::from_scalar(6, UINT16).unwrap(),
                ],
                FromVectorMode::Vector,
            )
            .unwrap();
            assert!(sub_v.is_equal(&tv_expected)?);

            let res = tv.get_sub_vector(None, Some(12), Some(1));
            assert!(res.is_err());

            let sub_v2 = tv.get_sub_vector(Some(1), None, Some(20))?;
            let tv_expected2 = TypedValue::from_vector(
                vec![TypedValue::from_scalar(1, UINT16).unwrap()],
                FromVectorMode::Vector,
            )
            .unwrap();
            assert!(sub_v2.is_equal(&tv_expected2)?);
            Ok(())
        }()
        .unwrap();
    }
}
