//! Definition of the [Value] struct and related functions, which handle data values within CipherCore.
use atomic_refcell::AtomicRefCell;

use std::convert::TryInto;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::ops::Not;
use std::sync::Arc;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::bytes::{vec_from_bytes, vec_to_bytes};
use crate::data_types::{array_type, get_size_in_bits, get_types_vector, ScalarType, Type, BIT};
use crate::errors::Result;

use crate::version::{VersionedData, DATA_VERSION};

#[derive(Clone, Debug, Serialize, Deserialize)]
enum SerializableValueBody {
    Bytes(Vec<u8>),
    Vector(Vec<SerializableValue>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SerializableValue {
    body: Arc<AtomicRefCellWrapper<SerializableValueBody>>,
}

impl SerializableValue {
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self {
            body: Arc::new(AtomicRefCellWrapper(AtomicRefCell::new(
                SerializableValueBody::Bytes(bytes),
            ))),
        }
    }

    pub fn from_vector(v: Vec<SerializableValue>) -> Self {
        Self {
            body: Arc::new(AtomicRefCellWrapper(AtomicRefCell::new(
                SerializableValueBody::Vector(v),
            ))),
        }
    }

    pub fn access<FB, FV, R>(&self, f_bytes: FB, f_vector: FV) -> Result<R>
    where
        FB: FnOnce(&[u8]) -> Result<R>,
        FV: FnOnce(&Vec<SerializableValue>) -> Result<R>,
    {
        match *self.body.0.borrow() {
            SerializableValueBody::Bytes(ref bytes) => f_bytes(bytes),
            SerializableValueBody::Vector(ref v) => f_vector(v),
        }
    }

    pub fn from_value(value: &Value) -> Self {
        value
            .access(
                |bytes| Ok(SerializableValue::from_bytes(bytes.to_vec())),
                |vector| {
                    let mut v_new = vec![];
                    for item in vector {
                        v_new.push(SerializableValue::from_value(item));
                    }
                    Ok(SerializableValue::from_vector(v_new))
                },
            )
            .expect("Error during conversion from Value to SerializableValue!")
    }
}

/// Bytes are in the little-endian form
#[derive(Clone, Debug, PartialEq, Eq)]
enum ValueBody {
    Bytes(Vec<u8>),
    Vector(Vec<Value>),
}

impl Value {
    /// Creates a fully disjoint clone of `self` via recursive traversal.
    ///
    /// # Returns
    ///
    /// Clone of `self`
    pub fn deep_clone(&self) -> Value {
        self.access(
            |bytes| Ok(Value::from_bytes(bytes.to_vec())),
            |vector| {
                let mut v_new = vec![];
                for item in vector {
                    v_new.push(item.deep_clone());
                }
                Ok(Value::from_vector(v_new))
            },
        )
        .unwrap()
    }

    /// Hashes `self` via recursive traversal.
    ///
    /// # Arguments
    ///
    /// `state` - hasher used for hashing
    pub fn deep_hash<H: Hasher>(&self, state: &mut H) {
        // Can't use `self.access()` here,
        // since both branches require write access to `state`.
        match &*self.body.0.borrow() {
            ValueBody::Bytes(data) => {
                data.hash(state);
            }
            ValueBody::Vector(elements) => {
                for element in elements {
                    element.deep_hash(state);
                }
            }
        }
    }
}

#[derive(PartialEq, Eq, Debug)]
struct AtomicRefCellWrapper<T>(pub AtomicRefCell<T>);

impl<T: Serialize> Serialize for AtomicRefCellWrapper<T> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.borrow().serialize(serializer)
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for AtomicRefCellWrapper<T> {
    fn deserialize<D>(deserializer: D) -> std::result::Result<AtomicRefCellWrapper<T>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let val = Deserialize::deserialize(deserializer)?;
        Ok(AtomicRefCellWrapper(AtomicRefCell::new(val)))
    }
}

/// A structure that stores pointer to a value that corresponds to an input, output or an intermediate result of
/// a computation.
///
/// A value is:
///   * Either a vector of bytes (for scalars or arrays);
///   * Or a vector of pointers to other values (for vectors, tuples or named tuples).
///
/// Overall, a value can be seen as a rooted tree of byte vectors.
///
/// [Clone] trait duplicates the pointer, not the underlying value (see [Value::deep_clone] for deep cloning).
///
/// [PartialEq] trait performs the deep recursive comparison.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Value {
    body: Arc<AtomicRefCellWrapper<ValueBody>>,
}

impl Serialize for Value {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let versioned_value = self
            .to_versioned_data()
            .expect("Error during conversion from Value into VersionedData");
        versioned_value.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Value {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        let versioned_value = VersionedData::deserialize(deserializer)?;
        if !versioned_value.check_version(DATA_VERSION) {
            Err(runtime_error!(
                "Value version doesn't match the requirement"
            ))
            .map_err(serde::de::Error::custom)
        } else {
            let serializable_value =
                serde_json::from_str::<SerializableValue>(versioned_value.get_data_string())
                    .expect("Error during conversion from String to SerializableValue!");
            Value::from_serializable_value(serializable_value).map_err(serde::de::Error::custom)
        }
    }
}

impl Value {
    /// Constructs a value from a given byte buffer.
    ///
    /// Not recommended to be used directly (instead, please use higher-level wrappers such as [Value::from_scalar]).
    ///
    /// # Arguments
    ///
    /// `bytes` - byte vector
    ///
    /// # Returns
    ///
    /// New value
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self {
            body: Arc::new(AtomicRefCellWrapper(AtomicRefCell::new(ValueBody::Bytes(
                bytes,
            )))),
        }
    }

    /// Constructs a value from a vector of other values.
    ///
    /// # Arguments
    ///
    /// `v` - vector of values
    ///
    /// # Returns
    ///
    /// New value constructed from `v`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// let v = Value::from_vector(
    ///     vec![
    ///         Value::from_bytes(vec![1, 2, 3]),
    ///         Value::from_bytes(vec![4, 5, 6]),
    ///         Value::from_vector(vec![])]);
    /// ```
    pub fn from_vector(v: Vec<Value>) -> Self {
        Self {
            body: Arc::new(AtomicRefCellWrapper(AtomicRefCell::new(ValueBody::Vector(
                v,
            )))),
        }
    }

    /// Constructs a value from a given bit or integer scalar.
    ///
    /// # Arguments
    ///
    /// * `x` - scalar to be converted to a value, can be of any standard integer type
    /// * `st` - scalar type corresponding to `x`
    ///
    /// # Returns
    ///
    /// New value constructed from `x`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::INT32;
    /// let v = Value::from_scalar(-123456, INT32).unwrap();
    /// v.access_bytes(|bytes| {
    ///     assert_eq!(*bytes, vec![192, 29, 254, 255]);
    ///     Ok(())
    /// }).unwrap();
    /// ```
    pub fn from_scalar<T: TryInto<u64> + Not<Output = T> + TryInto<u8> + Copy>(
        x: T,
        st: ScalarType,
    ) -> Result<Self> {
        let v = vec![x];
        Ok(Value::from_bytes(vec_to_bytes(&v, st)?))
    }

    /// Constructs a value from a flattened bit or integer array.
    ///
    /// # Arguments
    ///
    /// * `x` - array to be converted to a value, can have entries of any standard integer type
    /// * `st` - scalar type corresponding to the entries of `x`
    ///
    /// # Returns
    ///
    /// New value constructed from `x`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::BIT;
    /// let v = Value::from_flattened_array(&[0, 1, 1, 0, 1, 0, 0, 1], BIT).unwrap();
    /// v.access_bytes(|bytes| {
    ///     assert_eq!(*bytes, vec![150]);
    ///     Ok(())
    /// }).unwrap();
    /// ```
    pub fn from_flattened_array<T: TryInto<u64> + Not<Output = T> + TryInto<u8> + Copy>(
        x: &[T],
        st: ScalarType,
    ) -> Result<Value> {
        Ok(Value::from_bytes(vec_to_bytes(x, st)?))
    }

    /// Constructs a value from a multi-dimensional bit or integer array.
    ///
    /// # Arguments
    ///
    /// * `x` - array to be converted to a value, can have entries of any standard integer type
    /// * `st` - scalar type corresponding to the entries of `x`
    ///
    /// # Returns
    ///
    /// New value constructed from `x`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::BIT;
    /// # use ndarray::array;
    /// let a = array![[0, 1, 1, 0], [1, 0, 0, 1]].into_dyn();
    /// let v = Value::from_ndarray(a, BIT).unwrap();
    /// v.access_bytes(|bytes| {
    ///     assert_eq!(*bytes, vec![150]);
    ///     Ok(())
    /// }).unwrap();
    /// ```
    pub fn from_ndarray<T: TryInto<u64> + Not<Output = T> + TryInto<u8> + Copy>(
        a: ndarray::ArrayD<T>,
        st: ScalarType,
    ) -> Result<Value> {
        match a.as_slice() {
            Some(x) => Value::from_flattened_array(x, st),
            None => Err(runtime_error!("Not a contiguous ndarray!")),
        }
    }

    /// Converts `self` to a scalar if it is a byte vector, then casts the result to `u8`.
    ///
    /// # Arguments
    ///
    /// `st` - scalar type used to interpret `self`
    ///
    /// # Result
    ///
    /// Resulting scalar cast to `u8`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::INT32;
    /// let v = Value::from_scalar(-123456, INT32).unwrap();
    /// assert_eq!(v.to_u8(INT32).unwrap(), -123456i32 as u8);
    /// ```
    pub fn to_u8(&self, st: ScalarType) -> Result<u8> {
        Ok(self.to_u64(st)? as u8)
    }

    /// Converts `self` to a scalar if it is a byte vector, then casts the result to `bool`.
    ///
    /// # Result
    ///
    /// Resulting scalar cast to `bool`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::UINT8;
    /// let v = Value::from_scalar(156, UINT8).unwrap();
    /// assert_eq!(v.to_bit().unwrap(), false);
    /// ```
    pub fn to_bit(&self) -> Result<bool> {
        Ok(self.to_flattened_array_u8(array_type(vec![1], BIT))?[0] != 0)
    }

    /// Converts `self` to a scalar if it is a byte vector, then casts the result to `i8`.
    ///
    /// # Arguments
    ///
    /// `st` - scalar type used to interpret `self`
    ///
    /// # Result
    ///
    /// Resulting scalar cast to `i8`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::INT32;
    /// let v = Value::from_scalar(-123456, INT32).unwrap();
    /// assert_eq!(v.to_i8(INT32).unwrap(), -123456i32 as i8);
    /// ```
    pub fn to_i8(&self, st: ScalarType) -> Result<i8> {
        Ok(self.to_u64(st)? as i8)
    }

    /// Converts `self` to a scalar if it is a byte vector, then casts the result to `u16`.
    ///
    /// # Arguments
    ///
    /// `st` - scalar type used to interpret `self`
    ///
    /// # Result
    ///
    /// Resulting scalar cast to `u16`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::INT32;
    /// let v = Value::from_scalar(-123456, INT32).unwrap();
    /// assert_eq!(v.to_u16(INT32).unwrap(), -123456i32 as u16);
    /// ```
    pub fn to_u16(&self, st: ScalarType) -> Result<u16> {
        Ok(self.to_u64(st)? as u16)
    }

    /// Converts `self` to a scalar if it is a byte vector, then casts the result to `i16`.
    ///
    /// # Arguments
    ///
    /// `st` - scalar type used to interpret `self`
    ///
    /// # Result
    ///
    /// Resulting scalar cast to `i16`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::INT32;
    /// let v = Value::from_scalar(-123456, INT32).unwrap();
    /// assert_eq!(v.to_i16(INT32).unwrap(), -123456i32 as i16);
    /// ```
    pub fn to_i16(&self, st: ScalarType) -> Result<i16> {
        Ok(self.to_u64(st)? as i16)
    }

    /// Converts `self` to a scalar if it is a byte vector, then casts the result to `u32`.
    ///
    /// # Arguments
    ///
    /// `st` - scalar type used to interpret `self`
    ///
    /// # Result
    ///
    /// Resulting scalar cast to `u32`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::INT32;
    /// let v = Value::from_scalar(-123456, INT32).unwrap();
    /// assert_eq!(v.to_u32(INT32).unwrap(), -123456i32 as u32);
    /// ```
    pub fn to_u32(&self, st: ScalarType) -> Result<u32> {
        Ok(self.to_u64(st)? as u32)
    }

    /// Converts `self` to a scalar if it is a byte vector, then casts the result to `i32`.
    ///
    /// # Arguments
    ///
    /// `st` - scalar type used to interpret `self`
    ///
    /// # Result
    ///
    /// Resulting scalar cast to `i32`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::INT32;
    /// let v = Value::from_scalar(-123456, INT32).unwrap();
    /// assert_eq!(v.to_i32(INT32).unwrap(), -123456i32 as i32);
    /// ```
    pub fn to_i32(&self, st: ScalarType) -> Result<i32> {
        Ok(self.to_u64(st)? as i32)
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
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::INT32;
    /// let v = Value::from_scalar(-123456, INT32).unwrap();
    /// assert_eq!(v.to_u64(INT32).unwrap(), -123456i32 as u32 as u64);
    /// ```
    pub fn to_u64(&self, st: ScalarType) -> Result<u64> {
        let v = self.access_bytes(|bytes| vec_from_bytes(bytes, st.clone()))?;
        if v.len() != 1 && (v.len() != 8 || st != BIT) {
            return Err(runtime_error!("Not a scalar"));
        }
        Ok(v[0])
    }

    /// Converts `self` to a scalar if it is a byte vector, then cast the result to `i64`.
    ///
    /// # Arguments
    ///
    /// `st` - scalar type used to interpret `self`
    ///
    /// # Result
    ///
    /// Resulting scalar cast to `i64`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::INT32;
    /// let v = Value::from_scalar(-123456, INT32).unwrap();
    /// assert_eq!(v.to_i64(INT32).unwrap(), -123456i32 as u32 as i64);
    /// ```
    pub fn to_i64(&self, st: ScalarType) -> Result<i64> {
        Ok(self.to_u64(st)? as i64)
    }

    /// Converts `self` to a vector of values or return an error if `self` is a byte vector.
    ///
    /// # Returns
    ///
    /// Extracted vector of values
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// let v = Value::from_vector(
    ///     vec![
    ///         Value::from_vector(vec![]),
    ///         Value::from_bytes(vec![1, 2, 3])]);
    /// let vv = v.to_vector().unwrap();
    /// assert_eq!(vv.len(), 2);
    /// vv[0].access_vector(|v| {
    ///     assert_eq!(*v, Vec::<Value>::new());
    ///     Ok(())
    /// }).unwrap();
    /// vv[1].access_bytes(|bytes| {
    ///     assert_eq!(*bytes, vec![1, 2, 3]);
    ///     Ok(())
    /// }).unwrap();
    /// ```
    pub fn to_vector(&self) -> Result<Vec<Value>> {
        let cell = self.body.0.borrow();
        if let ValueBody::Vector(ref contents) = *cell {
            Ok(contents.clone())
        } else {
            Err(runtime_error!("Not a vector!"))
        }
    }

    /// Converts `self` to a flattened array if it is a byte vector, then cast the array entries to `u8`.
    ///
    /// # Arguments
    ///
    /// `t` - array type used to interpret `self`
    ///
    /// # Result
    ///
    /// Resulting flattened array with entries cast to `u8`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::{array_type, INT32};
    /// let v = Value::from_flattened_array(&[-123, 123], INT32).unwrap();
    /// let a = v.to_flattened_array_u8(array_type(vec![2], INT32)).unwrap();
    /// assert_eq!(a, vec![-123i32 as u8, 123i32 as u8]);
    /// ```
    pub fn to_flattened_array_u8(&self, t: Type) -> Result<Vec<u8>> {
        Ok(self
            .to_flattened_array_u64(t)?
            .into_iter()
            .map(|x| x as u8)
            .collect())
    }

    /// Converts `self` to a flattened array if it is a byte vector, then cast the array entries to `i8`.
    ///
    /// # Arguments
    ///
    /// `t` - array type used to interpret `self`
    ///
    /// # Result
    ///
    /// Resulting flattened array with entries cast to `i8`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::{array_type, INT32};
    /// let v = Value::from_flattened_array(&[-123, 123], INT32).unwrap();
    /// let a = v.to_flattened_array_i8(array_type(vec![2], INT32)).unwrap();
    /// assert_eq!(a, vec![-123i32 as i8, 123i32 as i8]);
    /// ```
    pub fn to_flattened_array_i8(&self, t: Type) -> Result<Vec<i8>> {
        Ok(self
            .to_flattened_array_u64(t)?
            .into_iter()
            .map(|x| x as i8)
            .collect())
    }

    /// Converts `self` to a flattened array if it is a byte vector, then cast the array entries to `u16`.
    ///
    /// # Arguments
    ///
    /// `t` - array type used to interpret `self`
    ///
    /// # Result
    ///
    /// Resulting flattened array with entries cast to `u16`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::{array_type, INT32};
    /// let v = Value::from_flattened_array(&[-123, 123], INT32).unwrap();
    /// let a = v.to_flattened_array_u16(array_type(vec![2], INT32)).unwrap();
    /// assert_eq!(a, vec![-123i32 as u16, 123i32 as u16]);
    /// ```
    pub fn to_flattened_array_u16(&self, t: Type) -> Result<Vec<u16>> {
        Ok(self
            .to_flattened_array_u64(t)?
            .into_iter()
            .map(|x| x as u16)
            .collect())
    }

    /// Converts `self` to a flattened array if it is a byte vector, then cast the array entries to `i16`.
    ///
    /// # Arguments
    ///
    /// `t` - array type used to interpret `self`
    ///
    /// # Result
    ///
    /// Resulting flattened array with entries cast to `i16`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::{array_type, INT32};
    /// let v = Value::from_flattened_array(&[-123, 123], INT32).unwrap();
    /// let a = v.to_flattened_array_i16(array_type(vec![2], INT32)).unwrap();
    /// assert_eq!(a, vec![-123i32 as i16, 123i32 as i16]);
    /// ```
    pub fn to_flattened_array_i16(&self, t: Type) -> Result<Vec<i16>> {
        Ok(self
            .to_flattened_array_u64(t)?
            .into_iter()
            .map(|x| x as i16)
            .collect())
    }

    /// Converts `self` to a flattened array if it is a byte vector, then cast the array entries to `u32`.
    ///
    /// # Arguments
    ///
    /// `t` - array type used to interpret `self`
    ///
    /// # Result
    ///
    /// Resulting flattened array with entries cast to `u32`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::{array_type, INT32};
    /// let v = Value::from_flattened_array(&[-123, 123], INT32).unwrap();
    /// let a = v.to_flattened_array_u32(array_type(vec![2], INT32)).unwrap();
    /// assert_eq!(a, vec![-123i32 as u32, 123i32 as u32]);
    /// ```
    pub fn to_flattened_array_u32(&self, t: Type) -> Result<Vec<u32>> {
        Ok(self
            .to_flattened_array_u64(t)?
            .into_iter()
            .map(|x| x as u32)
            .collect())
    }

    /// Converts `self` to a flattened array if it is a byte vector, then cast the array entries to `i32`.
    ///
    /// # Arguments
    ///
    /// `t` - array type used to interpret `self`
    ///
    /// # Result
    ///
    /// Resulting flattened array with entries cast to `i32`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::{array_type, INT32};
    /// let v = Value::from_flattened_array(&[-123, 123], INT32).unwrap();
    /// let a = v.to_flattened_array_i32(array_type(vec![2], INT32)).unwrap();
    /// assert_eq!(a, vec![-123i32, 123i32]);
    /// ```
    pub fn to_flattened_array_i32(&self, t: Type) -> Result<Vec<i32>> {
        Ok(self
            .to_flattened_array_u64(t)?
            .into_iter()
            .map(|x| x as i32)
            .collect())
    }

    /// Converts `self` to a flattened array if it is a byte vector, then cast the array entries to `u64`.
    ///
    /// # Arguments
    ///
    /// `t` - array type used to interpret `self`
    ///
    /// # Result
    ///
    /// Resulting flattened array with entries cast to `u64`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::{array_type, INT32};
    /// let v = Value::from_flattened_array(&[-123, 123], INT32).unwrap();
    /// let a = v.to_flattened_array_u64(array_type(vec![2], INT32)).unwrap();
    /// assert_eq!(a, vec![-123i32 as u32 as u64, 123i32 as u32 as u64]);
    /// ```
    pub fn to_flattened_array_u64(&self, t: Type) -> Result<Vec<u64>> {
        if !t.is_array() {
            return Err(runtime_error!(
                "Trying to extract array from a value of a wrong type"
            ));
        }
        if !self.check_type(t.clone())? {
            return Err(runtime_error!("Type and value mismatch"));
        }
        let deref_value = self.body.0.borrow().clone();
        let st = t.get_scalar_type();
        if let ValueBody::Bytes(ref bytes) = deref_value {
            let mut result = vec_from_bytes(bytes, st.clone())?;
            if st == BIT {
                let num_values: u64 = t.get_dimensions().iter().product();
                result.truncate(num_values as usize);
            }
            Ok(result)
        } else {
            Err(runtime_error!("Invalid Value"))
        }
    }

    /// Converts `self` to a flattened array if it is a byte vector, then cast the array entries to `i64`.
    ///
    /// # Arguments
    ///
    /// `t` - array type used to interpret `self`
    ///
    /// # Result
    ///
    /// Resulting flattened array with entries cast to `i64`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::{array_type, INT32};
    /// let v = Value::from_flattened_array(&[-123, 123], INT32).unwrap();
    /// let a = v.to_flattened_array_i64(array_type(vec![2], INT32)).unwrap();
    /// assert_eq!(a, vec![-123i32 as u32 as i64, 123i32 as u32 as i64]);
    /// ```
    pub fn to_flattened_array_i64(&self, t: Type) -> Result<Vec<i64>> {
        Ok(self
            .to_flattened_array_u64(t)?
            .into_iter()
            .map(|x| x as i64)
            .collect())
    }

    /// Converts `self` to a multi-dimensional array if it is a byte vector, then cast the array entries to `u8`.
    ///
    /// # Arguments
    ///
    /// `t` - array type used to interpret `self`
    ///
    /// # Result
    ///
    /// Resulting multi-dimensional array with entries cast to `u8`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// # use ndarray::array;
    /// let a = array![[-123, 123], [-456, 456]].into_dyn();
    /// let v = Value::from_ndarray(a, INT32).unwrap();
    /// let a = v.to_ndarray_u8(array_type(vec![2, 2], INT32)).unwrap();
    /// assert_eq!(a, array![[-123i32 as u8, 123i32 as u8], [-456i32 as u8, 456i32 as u8]].into_dyn());
    /// ```
    pub fn to_ndarray_u8(&self, t: Type) -> Result<ndarray::ArrayD<u8>> {
        let arr = self.to_ndarray_u64(t)?;
        Ok(arr.map(|x| *x as u8))
    }

    /// Converts `self` to a multi-dimensional array if it is a byte vector, then cast the array entries to `i8`.
    ///
    /// # Arguments
    ///
    /// `t` - array type used to interpret `self`
    ///
    /// # Result
    ///
    /// Resulting multi-dimensional array with entries cast to `i8`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// # use ndarray::array;
    /// let a = array![[-123, 123], [-456, 456]].into_dyn();
    /// let v = Value::from_ndarray(a, INT32).unwrap();
    /// let a = v.to_ndarray_i8(array_type(vec![2, 2], INT32)).unwrap();
    /// assert_eq!(a, array![[-123i32 as i8, 123i32 as i8], [-456i32 as i8, 456i32 as i8]].into_dyn());
    /// ```
    pub fn to_ndarray_i8(&self, t: Type) -> Result<ndarray::ArrayD<i8>> {
        let arr = self.to_ndarray_u64(t)?;
        Ok(arr.map(|x| *x as i8))
    }

    /// Converts `self` to a multi-dimensional array if it is a byte vector, then cast the array entries to `u16`.
    ///
    /// # Arguments
    ///
    /// `t` - array type used to interpret `self`
    ///
    /// # Result
    ///
    /// Resulting multi-dimensional array with entries cast to `u16`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// # use ndarray::array;
    /// let a = array![[-123, 123], [-456, 456]].into_dyn();
    /// let v = Value::from_ndarray(a, INT32).unwrap();
    /// let a = v.to_ndarray_u16(array_type(vec![2, 2], INT32)).unwrap();
    /// assert_eq!(a, array![[-123i32 as u16, 123i32 as u16], [-456i32 as u16, 456i32 as u16]].into_dyn());
    /// ```
    pub fn to_ndarray_u16(&self, t: Type) -> Result<ndarray::ArrayD<u16>> {
        let arr = self.to_ndarray_u64(t)?;
        Ok(arr.map(|x| *x as u16))
    }

    /// Converts `self` to a multi-dimensional array if it is a byte vector, then cast the array entries to `i16`.
    ///
    /// # Arguments
    ///
    /// `t` - array type used to interpret `self`
    ///
    /// # Result
    ///
    /// Resulting multi-dimensional array with entries cast to `i16`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// # use ndarray::array;
    /// let a = array![[-123, 123], [-456, 456]].into_dyn();
    /// let v = Value::from_ndarray(a, INT32).unwrap();
    /// let a = v.to_ndarray_i16(array_type(vec![2, 2], INT32)).unwrap();
    /// assert_eq!(a, array![[-123i32 as i16, 123i32 as i16], [-456i32 as i16, 456i32 as i16]].into_dyn());
    /// ```
    pub fn to_ndarray_i16(&self, t: Type) -> Result<ndarray::ArrayD<i16>> {
        let arr = self.to_ndarray_u64(t)?;
        Ok(arr.map(|x| *x as i16))
    }

    /// Converts `self` to a multi-dimensional array if it is a byte vector, then cast the array entries to `u32`.
    ///
    /// # Arguments
    ///
    /// `t` - array type used to interpret `self`
    ///
    /// # Result
    ///
    /// Resulting multi-dimensional array with entries cast to `u32`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// # use ndarray::array;
    /// let a = array![[-123, 123], [-456, 456]].into_dyn();
    /// let v = Value::from_ndarray(a, INT32).unwrap();
    /// let a = v.to_ndarray_u32(array_type(vec![2, 2], INT32)).unwrap();
    /// assert_eq!(a, array![[-123i32 as u32, 123i32 as u32], [-456i32 as u32, 456i32 as u32]].into_dyn());
    /// ```
    pub fn to_ndarray_u32(&self, t: Type) -> Result<ndarray::ArrayD<u32>> {
        let arr = self.to_ndarray_u64(t)?;
        Ok(arr.map(|x| *x as u32))
    }

    /// Converts `self` to a multi-dimensional array if it is a byte vector, then cast the array entries to `i32`.
    ///
    /// # Arguments
    ///
    /// `t` - array type used to interpret `self`
    ///
    /// # Result
    ///
    /// Resulting multi-dimensional array with entries cast to `i32`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// # use ndarray::array;
    /// let a = array![[-123, 123], [-456, 456]].into_dyn();
    /// let v = Value::from_ndarray(a, INT32).unwrap();
    /// let a = v.to_ndarray_i32(array_type(vec![2, 2], INT32)).unwrap();
    /// assert_eq!(a, array![[-123i32, 123i32], [-456i32, 456i32]].into_dyn());
    /// ```
    pub fn to_ndarray_i32(&self, t: Type) -> Result<ndarray::ArrayD<i32>> {
        let arr = self.to_ndarray_u64(t)?;
        Ok(arr.map(|x| *x as i32))
    }

    /// Converts `self` to a multi-dimensional array if it is a byte vector, then cast the array entries to `u64`.
    ///
    /// # Arguments
    ///
    /// `t` - array type used to interpret `self`
    ///
    /// # Result
    ///
    /// Resulting multi-dimensional array with entries cast to `u64`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// # use ndarray::array;
    /// let a = array![[-123, 123], [-456, 456]].into_dyn();
    /// let v = Value::from_ndarray(a, INT32).unwrap();
    /// let a = v.to_ndarray_u64(array_type(vec![2, 2], INT32)).unwrap();
    /// assert_eq!(a, array![[-123i32 as u32 as u64, 123i32 as u32 as u64], [-456i32 as u32 as u64, 456i32 as u32 as u64]].into_dyn());
    /// ```
    pub fn to_ndarray_u64(&self, t: Type) -> Result<ndarray::ArrayD<u64>> {
        match t.clone() {
            Type::Array(shape, _) => {
                let arr = self.to_flattened_array_u64(t)?;
                // TODO: for performance reasons, we should use the actual type, not u64.
                let ndarr = ndarray::Array::from_vec(arr);
                let shape: Vec<usize> = shape.iter().map(|x| *x as usize).collect();
                Ok(ndarr.into_shape(shape)?)
            }
            _ => Err(runtime_error!("Not an array type")),
        }
    }

    /// Converts `self` to a multi-dimensional array if it is a byte vector, then cast the array entries to `u64`.
    ///
    /// # Arguments
    ///
    /// `t` - array type used to interpret `self`
    ///
    /// # Result
    ///
    /// Resulting multi-dimensional array with entries cast to `u64`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::{BIT, array_type};
    /// # use ndarray::array;
    /// let a = array![[false, true], [true, false]].into_dyn();
    /// let v = Value::from_ndarray(a.clone(), BIT).unwrap();
    /// let converted = v.to_ndarray_bool(array_type(vec![2, 2], BIT)).unwrap();
    /// assert_eq!(converted, a);
    /// ```
    pub fn to_ndarray_bool(&self, t: Type) -> Result<ndarray::ArrayD<bool>> {
        match t.clone() {
            Type::Array(shape, _) => {
                let arr = self
                    .to_flattened_array_u8(t)?
                    .iter()
                    .map(|x| *x != 0)
                    .collect();
                let ndarr = ndarray::Array::from_vec(arr);
                let shape: Vec<usize> = shape.iter().map(|x| *x as usize).collect();
                Ok(ndarr.into_shape(shape)?)
            }
            _ => Err(runtime_error!("Not an array type")),
        }
    }

    /// Converts `self` to a multi-dimensional array if it is a byte vector, then cast the array entries to `i64`.
    ///
    /// # Arguments
    ///
    /// `t` - array type used to interpret `self`
    ///
    /// # Result
    ///
    /// Resulting multi-dimensional array with entries cast to `i64`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::{INT32, array_type};
    /// # use ndarray::array;
    /// let a = array![[-123, 123], [-456, 456]].into_dyn();
    /// let v = Value::from_ndarray(a, INT32).unwrap();
    /// let a = v.to_ndarray_i64(array_type(vec![2, 2], INT32)).unwrap();
    /// assert_eq!(a, array![[-123i32 as u32 as i64, 123i32 as u32 as i64], [-456i32 as u32 as i64, 456i32 as u32 as i64]].into_dyn());
    /// ```
    pub fn to_ndarray_i64(&self, t: Type) -> Result<ndarray::ArrayD<i64>> {
        let arr = self.to_ndarray_u64(t)?;
        Ok(arr.map(|x| *x as i64))
    }

    /// Checks if `self` is a valid value for a given type.
    ///
    /// # Arguments
    ///
    /// `t` - a type to check a value against
    ///
    /// # Returns
    ///
    /// `true` if `self` is a valid value of type `t`, `false` otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::{scalar_type, INT32, array_type, UINT32, UINT8, BIT, UINT16, UINT64};
    /// assert!(Value::from_bytes(vec![1, 2, 3, 4]).check_type(scalar_type(INT32)).unwrap());
    /// assert!(Value::from_bytes(vec![1, 2, 3, 4]).check_type(scalar_type(UINT32)).unwrap());
    /// assert!(Value::from_bytes(vec![1, 2, 3, 4]).check_type(array_type(vec![2, 2], UINT8)).unwrap());
    /// assert!(Value::from_bytes(vec![1, 2, 3, 4]).check_type(array_type(vec![4, 8], BIT)).unwrap());
    /// assert!(!Value::from_bytes(vec![1, 2, 3, 4]).check_type(array_type(vec![3, 5], BIT)).unwrap());
    /// assert!(!Value::from_bytes(vec![1, 2, 3, 4]).check_type(array_type(vec![5], UINT8)).unwrap());
    /// assert!(!Value::from_bytes(vec![1, 2, 3, 4]).check_type(array_type(vec![3], UINT16)).unwrap());
    /// assert!(!Value::from_bytes(vec![1, 2, 3, 4]).check_type(scalar_type(UINT64)).unwrap());
    /// ```
    pub fn check_type(&self, t: Type) -> Result<bool> {
        let s = get_size_in_bits(t.clone())?;
        match t {
            Type::Scalar(_) | Type::Array(_, _) => match *self.body.0.borrow() {
                ValueBody::Bytes(ref bytes) => Ok(bytes.len() as u64 == (s + 7) / 8),
                _ => Ok(false),
            },
            Type::Vector(_, _) | Type::Tuple(_) | Type::NamedTuple(_) => {
                let ts = get_types_vector(t)?;
                match *self.body.0.borrow() {
                    ValueBody::Vector(ref children) => {
                        if ts.len() != children.len() {
                            return Ok(false);
                        }
                        for i in 0..ts.len() {
                            if !children[i].check_type((*ts[i]).clone())? {
                                return Ok(false);
                            }
                        }
                        Ok(true)
                    }
                    _ => Ok(false),
                }
            }
        }
    }

    /// Runs a given closure if `self` corresponds to a byte vector, and panic otherwise.
    ///
    /// # Arguments
    ///
    /// `f` - a closure, that takes a reference to a slice of bytes, to run
    ///
    /// # Returns
    ///
    /// Return value of `f`
    ///
    /// # Panics
    ///
    /// Panics if `self` is a vector of values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    ///
    /// let v = Value::from_bytes(vec![1, 2, 3]);
    /// v.access_bytes(|bytes| {
    ///     assert_eq!(*bytes, vec![1, 2, 3]);
    ///     Ok(())
    /// }).unwrap();
    /// ```
    pub fn access_bytes<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&[u8]) -> Result<R>,
    {
        if let ValueBody::Bytes(ref bytes) = *self.body.0.borrow() {
            f(bytes)
        } else {
            panic!("Value::access_bytes() on an invalid Value");
        }
    }

    /// Runs a given closure if `self` corresponds to a vector of values, and panic otherwise.
    ///
    /// # Arguments
    ///
    /// `f` - a closure, that takes a reference to a vector of values, to run
    ///
    /// # Returns
    ///
    /// Return value of `f`
    ///
    /// # Panics
    ///
    /// Panics if `self` is a byte vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    ///
    /// let v = Value::from_vector(vec![]);
    /// v.access_vector(|vector| {
    ///     assert!(vector.is_empty());
    ///     Ok(())
    /// }).unwrap();
    /// ```
    pub fn access_vector<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&Vec<Value>) -> Result<R>,
    {
        if let ValueBody::Vector(ref v) = *self.body.0.borrow() {
            f(v)
        } else {
            panic!("Value::access_vector() on an invalid Value");
        }
    }

    /// Runs one closure if `self` corresponds to a byte vector,
    /// and another closure if `self` corresponds to a vector of values.
    ///
    /// # Arguments
    ///
    /// * `f_bytes` - a closure, that takes a reference to a slice of bytes, to run if `self` corresponds to a byte vector
    /// * `f_vector` - a closure, that takes a reference to a vector of values, to run if `self` corresponds to a vector of values
    ///
    /// # Returns
    ///
    /// Return value of the called closure
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    ///
    /// let v1 = Value::from_vector(vec![]);
    /// let v2 = Value::from_bytes(vec![1, 2, 3]);
    /// v1.access(|bytes| {
    ///     assert!(false);
    ///     Ok(())
    /// },
    /// |vector| {
    ///     assert!(vector.is_empty());
    ///     Ok(())
    /// }).unwrap();
    /// v2.access(|bytes| {
    ///     assert_eq!(*bytes, vec![1, 2, 3]);
    ///     Ok(())
    /// },
    /// |vector| {
    ///     assert!(false);
    ///     Ok(())
    /// }).unwrap();
    /// ```
    pub fn access<FB, FV, R>(&self, f_bytes: FB, f_vector: FV) -> Result<R>
    where
        FB: FnOnce(&[u8]) -> Result<R>,
        FV: FnOnce(&Vec<Value>) -> Result<R>,
    {
        match *self.body.0.borrow() {
            ValueBody::Bytes(ref bytes) => f_bytes(bytes),
            ValueBody::Vector(ref v) => f_vector(v),
        }
    }

    /// Generates a value of a given type with all-zero bytes.
    ///
    /// # Arguments
    ///
    /// `t` - the type of a new value
    ///
    /// # Returns
    ///
    /// "Zero" value of type `t`
    ///
    /// # Examples
    ///
    /// ```
    /// # use ciphercore_base::data_values::Value;
    /// # use ciphercore_base::data_types::{array_type, INT32};
    /// # use ndarray::array;
    /// let v = Value::zero_of_type(array_type(vec![2, 2], INT32));
    /// let a = v.to_ndarray_i32(array_type(vec![2, 2], INT32)).unwrap();
    /// assert_eq!(a, array![[0, 0], [0, 0]].into_dyn());
    /// ```
    pub fn zero_of_type(t: Type) -> Value {
        match t {
            Type::Scalar(_) | Type::Array(_, _) => {
                let s = get_size_in_bits(t.clone()).unwrap();
                Value::from_bytes(vec![0; ((s + 7) / 8) as usize])
            }
            Type::Vector(len, t1) => {
                Value::from_vector(vec![Value::zero_of_type((*t1).clone()); len as usize])
            }
            Type::Tuple(element_types) => Value::from_vector(
                element_types
                    .iter()
                    .map(|t| Value::zero_of_type((**t).clone()))
                    .collect(),
            ),
            Type::NamedTuple(element_types) => Value::from_vector(
                element_types
                    .iter()
                    .map(|(_, t)| Value::zero_of_type((**t).clone()))
                    .collect(),
            ),
        }
    }

    fn from_serializable_value(value: SerializableValue) -> Result<Value> {
        value.access(
            |bytes| Ok(Value::from_bytes(bytes.to_vec())),
            |vector| {
                let mut v_new = vec![];
                for item in vector {
                    v_new.push(Value::from_serializable_value(item.clone())?);
                }
                Ok(Value::from_vector(v_new))
            },
        )
    }

    fn to_versioned_data(&self) -> Result<VersionedData> {
        VersionedData::create_versioned_data(
            DATA_VERSION,
            serde_json::to_string(&SerializableValue::from_value(self))?,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants;
    use crate::data_types::{
        array_type, create_scalar_type, named_tuple_type, scalar_type, tuple_type, vector_type,
        BIT, INT32, INT64, INT8, UINT16, UINT32, UINT64, UINT8,
    };
    use std::panic::{catch_unwind, AssertUnwindSafe};

    fn check_type_test_worker(v: &Value, t: Type) {
        assert!(v.check_type(t).unwrap());
    }

    fn check_type_test_worker_fail(v: &Value, t: Type) {
        assert!(!v.check_type(t).unwrap());
    }

    #[test]
    fn check_type_test() {
        let v = Value::from_bytes(vec![0]);
        check_type_test_worker(&v, scalar_type(BIT));
        check_type_test_worker(&v, scalar_type(UINT8));
        check_type_test_worker(&v, scalar_type(INT8));
        check_type_test_worker_fail(&v, scalar_type(INT32));
        let v = Value::from_bytes(vec![0, 0, 0, 0]);
        check_type_test_worker(&v, scalar_type(INT32));
        check_type_test_worker_fail(&v, scalar_type(INT64));
        let v = Value::from_bytes(vec![0, 0, 0, 0, 0, 0, 0, 0]);
        check_type_test_worker(&v, scalar_type(INT64));
        let v = Value::from_vector(vec![]);
        check_type_test_worker_fail(&v, scalar_type(BIT));
        let v = Value::from_vector(vec![
            Value::from_bytes(vec![0]),
            Value::from_bytes(vec![0, 0, 0, 0]),
        ]);
        check_type_test_worker(&v, tuple_type(vec![scalar_type(BIT), scalar_type(INT32)]));
        check_type_test_worker(
            &v,
            named_tuple_type(vec![
                ("field 1".to_owned(), scalar_type(BIT)),
                ("field 2".to_owned(), scalar_type(INT32)),
            ]),
        );
        let v = Value::from_vector(vec![Value::from_bytes(vec![0]), Value::from_bytes(vec![0])]);
        check_type_test_worker(&v, vector_type(2, scalar_type(BIT)));
        check_type_test_worker_fail(&v, tuple_type(vec![]));
        let v = Value::from_bytes(vec![0, 0, 0]);
        check_type_test_worker_fail(&v, tuple_type(vec![]));
        if constants::NON_STANDARD_SCALAR_LEN_SUPPORT {
            let v = Value::from_bytes(vec![0]);
            check_type_test_worker(&v, scalar_type(create_scalar_type(false, Some(253))));
            check_type_test_worker(&v, scalar_type(create_scalar_type(false, Some(254))));
            check_type_test_worker(&v, scalar_type(create_scalar_type(false, Some(255))));
            check_type_test_worker_fail(&v, scalar_type(create_scalar_type(false, Some(257))));
            check_type_test_worker_fail(&v, scalar_type(create_scalar_type(false, Some(258))));
            check_type_test_worker_fail(&v, scalar_type(create_scalar_type(false, Some(259))));
            let v = Value::from_bytes(vec![0, 0]);
            check_type_test_worker(&v, scalar_type(create_scalar_type(false, Some(257))));
            check_type_test_worker(&v, scalar_type(create_scalar_type(false, Some(258))));
            check_type_test_worker(&v, scalar_type(create_scalar_type(false, Some(259))));
            let v = Value::from_bytes(vec![0, 0, 0, 0, 0, 0, 0, 0]);
            check_type_test_worker(
                &v,
                scalar_type(create_scalar_type(false, Some(9223372036854775808))),
            );
            check_type_test_worker(
                &v,
                scalar_type(create_scalar_type(false, Some(9223372036854775809))),
            );
            check_type_test_worker(
                &v,
                scalar_type(create_scalar_type(false, Some(9223372036854775810))),
            );
            let v = Value::from_bytes(vec![0, 0, 0, 0, 0, 0, 0]);
            check_type_test_worker(
                &v,
                array_type(vec![8], create_scalar_type(false, Some(128))),
            );
            check_type_test_worker_fail(
                &v,
                array_type(vec![9], create_scalar_type(false, Some(128))),
            );
            let v = Value::from_bytes(vec![0, 0, 0, 0, 0, 0, 0, 0]);
            check_type_test_worker(
                &v,
                array_type(vec![9], create_scalar_type(false, Some(128))),
            );
            check_type_test_worker_fail(
                &v,
                array_type(vec![8], create_scalar_type(false, Some(128))),
            );
        }
    }

    #[test]
    fn eq_test() {
        let a = Value::from_bytes(vec![10, 10]);
        let b = Value::from_bytes(vec![10, 10]);
        assert_eq!(a, b);
        let a = Value::from_vector(vec![Value::from_bytes(vec![10])]);
        let b = Value::from_vector(vec![Value::from_bytes(vec![10])]);
        assert_eq!(a, b);
        let a = Value::from_vector(vec![
            Value::from_bytes(vec![10]),
            Value::from_bytes(vec![7]),
        ]);
        let b = Value::from_vector(vec![
            Value::from_bytes(vec![10]),
            Value::from_bytes(vec![10]),
        ]);
        assert!(a != b);
        let a = Value::from_vector(vec![
            Value::from_bytes(vec![10]),
            Value::from_bytes(vec![7]),
        ]);
        let b = Value::from_bytes(vec![10, 10]);
        assert!(a != b);
        assert!(b != a);
        let a = Value::from_vector(vec![Value::from_bytes(vec![10])]);
        let b = Value::from_vector(vec![
            Value::from_bytes(vec![10]),
            Value::from_bytes(vec![10]),
        ]);
        assert!(a != b);
    }

    #[test]
    fn test_get_bytes() {
        let v = Value::from_bytes(vec![0, 1, 2, 3]);
        v.access_bytes(|bytes| {
            assert_eq!(bytes, vec![0, 1, 2, 3]);
            Ok(())
        })
        .unwrap();

        let v = Value::from_vector(vec![Value::from_bytes(vec![0]), Value::from_bytes(vec![0])]);
        let e = catch_unwind(AssertUnwindSafe(|| v.access_bytes(|_| Ok(()))));
        assert!(e.is_err());
    }

    #[test]
    fn test_serialization() {
        let v = Value::from_vector(vec![
            Value::from_bytes(vec![1, 2, 3, 4, 5]),
            Value::from_bytes(vec![6, 7, 8]),
        ]);
        let se = serde_json::to_string(&v).unwrap();
        assert_eq!(
            se,
            format!("{{\"version\":{},\"data\":\"{{\\\"body\\\":{{\\\"Vector\\\":[{{\\\"body\\\":{{\\\"Bytes\\\":[1,2,3,4,5]}}}},{{\\\"body\\\":{{\\\"Bytes\\\":[6,7,8]}}}}]}}}}\"}}", DATA_VERSION)
        );
        let de: Value = serde_json::from_str(&se).unwrap();
        assert_eq!(v, de);
        assert!(serde_json::from_str::<Value>("{{{").is_err());
    }

    #[test]
    #[should_panic(expected = "Value version doesn't match the requirement")]
    fn test_version_value() {
        let invalid_versioned_value = VersionedData::create_versioned_data(
            DATA_VERSION - 1,
            serde_json::to_string(&SerializableValue::from_vector(vec![
                SerializableValue::from_bytes(vec![1, 2, 3, 4, 5]),
                SerializableValue::from_bytes(vec![6, 7, 8]),
            ]))
            .unwrap(),
        )
        .unwrap();
        let se = serde_json::to_string(&invalid_versioned_value).unwrap();
        let _de: Value = serde_json::from_str(&se).unwrap();
    }

    #[test]
    #[should_panic(expected = "Value version doesn't match the requirement")]
    fn test_unsupported_version_value() {
        let invalid_versioned_value = VersionedData::create_versioned_data(
            DATA_VERSION - 1,
            "{\"Unsupported_field\": Unsupported_value}".to_string(),
        )
        .unwrap();
        let se = serde_json::to_string(&invalid_versioned_value).unwrap();
        let _de: Value = serde_json::from_str(&se).unwrap();
    }

    #[test]
    fn test_extract_scalar_bit() {
        let v = Value::from_scalar(1, BIT).unwrap();
        let result = v.to_u64(BIT).unwrap();
        assert_eq!(result, 1);
    }

    #[test]
    fn test_create_extract_scalar() {
        assert_eq!(Value::from_scalar(0, BIT).unwrap().to_u64(BIT).unwrap(), 0);
        assert_eq!(Value::from_scalar(1, BIT).unwrap().to_u64(BIT).unwrap(), 1);
        assert_eq!(
            Value::from_scalar(-73, INT32)
                .unwrap()
                .to_i32(INT32)
                .unwrap(),
            -73
        );
        assert_eq!(
            Value::from_scalar(-73, INT32)
                .unwrap()
                .to_i32(UINT32)
                .unwrap(),
            -73
        );
        assert_eq!(
            Value::from_scalar(187263, INT32)
                .unwrap()
                .to_i32(INT32)
                .unwrap(),
            187263
        );
        assert_eq!(
            Value::from_scalar(187263, UINT32)
                .unwrap()
                .to_u32(UINT32)
                .unwrap(),
            187263
        );
        assert_eq!(
            Value::from_flattened_array(&vec![0, 0, 0, 0, 0, 0, 0, 0], BIT)
                .unwrap()
                .to_u64(BIT)
                .unwrap(),
            0
        );
        assert_eq!(
            Value::from_flattened_array(&vec![1, 0, 0, 0, 0, 0, 0, 0], BIT)
                .unwrap()
                .to_u64(BIT)
                .unwrap(),
            1
        );
        assert_eq!(
            Value::from_flattened_array(&vec![1, 0, 0, 1, 0, 1, 0, 0], BIT)
                .unwrap()
                .to_u64(BIT)
                .unwrap(),
            1
        );
        assert!(
            Value::from_flattened_array(&vec![0, 0, 0, 0, 0, 0, 0, 0, 0], BIT)
                .unwrap()
                .to_u64(BIT)
                .is_err()
        );
        assert!(Value::from_flattened_array(&vec![123, 456], UINT32)
            .unwrap()
            .to_u64(UINT32)
            .is_err());
    }

    #[test]
    fn test_create_extract_vector() {
        let v = Value::from_vector(vec![
            Value::from_scalar(0, BIT).unwrap(),
            Value::from_scalar(-73, INT32).unwrap(),
        ]);
        let entries = v.to_vector().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].to_u64(BIT).unwrap(), 0);
        assert_eq!(entries[1].to_i32(INT32).unwrap(), -73);
        assert!(Value::from_scalar(-73, INT32).unwrap().to_vector().is_err());
    }

    #[test]
    fn test_create_extract_array() {
        let t = array_type(vec![1], BIT);
        assert_eq!(
            Value::from_flattened_array(&vec![0], t.get_scalar_type())
                .unwrap()
                .to_flattened_array_u64(t.clone())
                .unwrap(),
            vec![0]
        );
        assert_eq!(
            Value::from_flattened_array(&vec![1], t.get_scalar_type())
                .unwrap()
                .to_flattened_array_u64(t.clone())
                .unwrap(),
            vec![1]
        );
        assert_eq!(
            Value::from_flattened_array(&vec![1, 0, 0, 1], t.get_scalar_type())
                .unwrap()
                .to_flattened_array_u64(t.clone())
                .unwrap(),
            vec![1]
        );
        assert_eq!(
            Value::from_flattened_array(&vec![0, 0, 0, 1, 0, 0, 0, 0], t.get_scalar_type())
                .unwrap()
                .to_flattened_array_u64(t.clone())
                .unwrap(),
            vec![0]
        );
        assert!(Value::from_flattened_array(
            &vec![0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
            t.get_scalar_type()
        )
        .unwrap()
        .to_flattened_array_u64(t.clone())
        .is_err());
        let t = array_type(vec![3, 3], BIT);
        assert_eq!(
            Value::from_flattened_array(&vec![0, 1, 1, 0, 1, 0, 0, 1, 0], t.get_scalar_type())
                .unwrap()
                .to_flattened_array_u64(t.clone())
                .unwrap(),
            vec![0, 1, 1, 0, 1, 0, 0, 1, 0]
        );
        assert!(
            Value::from_flattened_array(&vec![0, 1, 1, 0, 1, 0, 0, 1], t.get_scalar_type())
                .unwrap()
                .to_flattened_array_u64(t.clone())
                .is_err()
        );
        assert!(Value::from_flattened_array(
            &vec![0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            t.get_scalar_type()
        )
        .unwrap()
        .to_flattened_array_u64(t.clone())
        .is_err());
        let t = array_type(vec![2, 3], INT32);
        assert_eq!(
            Value::from_flattened_array(&vec![1, 2, 3, 4, 5, 6], t.get_scalar_type())
                .unwrap()
                .to_flattened_array_u64(t.clone())
                .unwrap(),
            vec![1, 2, 3, 4, 5, 6]
        );
        assert!(
            Value::from_flattened_array(&vec![1, 2, 3, 4, 5, 6, 7], t.get_scalar_type())
                .unwrap()
                .to_flattened_array_u64(t.clone())
                .is_err()
        );
        assert!(
            Value::from_flattened_array(&vec![1, 2, 3, 4, 5], t.get_scalar_type())
                .unwrap()
                .to_flattened_array_u64(t.clone())
                .is_err()
        );
    }

    #[test]
    fn test_zero_value_of_type() {
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
            assert!(Value::zero_of_type(t.clone())
                .check_type(t.clone())
                .unwrap());
        }
    }
    #[test]
    fn test_get_types_vector() {
        let t = vector_type(100, scalar_type(UINT16));
        let result = get_types_vector(t);
        assert!(result.is_ok());
        let t = vector_type(
            constants::TYPES_VECTOR_LENGTH_LIMIT as u64 + 1,
            scalar_type(UINT16),
        );
        let result = get_types_vector(t);
        assert!(result.is_err());
    }

    #[test]
    fn test_deep_clone() {
        || -> Result<()> {
            let v = Value::from_vector(vec![
                Value::from_scalar(123, INT64)?,
                Value::from_flattened_array(&[-1, 1, -2, 2], INT64)?,
            ]);
            let v1 = v.deep_clone();
            assert!(!Arc::ptr_eq(&v.body, &v1.body));
            let vv = v.to_vector()?;
            let vv1 = v1.to_vector()?;
            assert_eq!(vv.len(), 2);
            assert_eq!(vv1.len(), 2);
            assert!(!Arc::ptr_eq(&vv[0].body, &vv1[0].body));
            assert!(!Arc::ptr_eq(&vv[1].body, &vv1[1].body));
            let vv_a = vv[0].to_i64(INT64)?;
            let vv_b = vv[1].to_flattened_array_i64(array_type(vec![2, 2], INT64))?;
            let vv1_a = vv1[0].to_i64(INT64)?;
            let vv1_b = vv1[1].to_flattened_array_i64(array_type(vec![2, 2], INT64))?;
            assert_eq!(vv_a, 123);
            assert_eq!(vv_b, &[-1, 1, -2, 2]);
            assert_eq!(vv1_a, 123);
            assert_eq!(vv1_b, &[-1, 1, -2, 2]);
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_from_ndarray() {
        || -> Result<()> {
            let a = ndarray::Array::from_shape_vec((2, 3), vec![10, -20, 30, 40, -50, 60])?;
            let v = Value::from_ndarray(a.into_dyn(), INT32)?;
            let b = v.to_flattened_array_i32(array_type(vec![2, 3], INT32))?;
            assert_eq!(b, &[10, -20, 30, 40, -50, 60]);
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_to_scalar() {
        || -> Result<()> {
            let v = Value::from_scalar(-123456, INT32)?;
            assert_eq!(v.to_u8(INT32)?, (-123456i32) as u8);
            assert_eq!(v.to_i8(INT32)?, (-123456i32) as i8);
            assert_eq!(v.to_u16(INT32)?, (-123456i32) as u16);
            assert_eq!(v.to_i16(INT32)?, (-123456i32) as i16);
            assert_eq!(v.to_u32(INT32)?, (-123456i32) as u32);
            assert_eq!(v.to_i32(INT32)?, (-123456i32) as i32);
            assert_eq!(v.to_u64(INT32)?, 4294843840u64);
            assert_eq!(v.to_i64(INT32)?, 4294843840i64);

            assert_eq!(Value::from_scalar(156, UINT8)?.to_bit()?, false);
            assert_eq!(Value::from_scalar(157, UINT8)?.to_bit()?, true);
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_to_flattened_array() {
        || -> Result<()> {
            let v = Value::from_scalar(-123456, INT32)?;
            assert_eq!(
                v.to_flattened_array_u8(array_type(vec![1], INT32))?,
                &[(-123456i32) as u8]
            );
            assert_eq!(
                v.to_flattened_array_i8(array_type(vec![1], INT32))?,
                &[(-123456i32) as i8]
            );
            assert_eq!(
                v.to_flattened_array_u16(array_type(vec![1], INT32))?,
                &[(-123456i32) as u16]
            );
            assert_eq!(
                v.to_flattened_array_i16(array_type(vec![1], INT32))?,
                &[(-123456i32) as i16]
            );
            assert_eq!(
                v.to_flattened_array_u32(array_type(vec![1], INT32))?,
                &[(-123456i32) as u32]
            );
            assert_eq!(
                v.to_flattened_array_i32(array_type(vec![1], INT32))?,
                &[(-123456i32) as i32]
            );
            assert_eq!(
                v.to_flattened_array_u64(array_type(vec![1], INT32))?,
                &[4294843840u64]
            );
            assert_eq!(
                v.to_flattened_array_i64(array_type(vec![1], INT32))?,
                &[4294843840i64]
            );
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_to_ndarray() {
        || -> Result<()> {
            {
                let v = Value::from_scalar(1, BIT)?;
                let a = v.to_ndarray_bool(array_type(vec![1], BIT))?;
                assert_eq!(a.shape(), &[1]);
                assert_eq!(a[[0]], true);
            }
            {
                let v = Value::from_scalar(0, BIT)?;
                let a = v.to_ndarray_bool(array_type(vec![1], BIT))?;
                assert_eq!(a.shape(), &[1]);
                assert_eq!(a[[0]], false);
            }
            let v = Value::from_scalar(-123456, INT32)?;
            {
                let a = v.to_ndarray_u8(array_type(vec![1], INT32))?;
                assert_eq!(a.shape(), &[1]);
                assert_eq!(a[[0]], -123456i32 as u8);
            }
            {
                let a = v.to_ndarray_i8(array_type(vec![1], INT32))?;
                assert_eq!(a.shape(), &[1]);
                assert_eq!(a[[0]], -123456i32 as i8);
            }
            {
                let a = v.to_ndarray_u16(array_type(vec![1], INT32))?;
                assert_eq!(a.shape(), &[1]);
                assert_eq!(a[[0]], -123456i32 as u16);
            }
            {
                let a = v.to_ndarray_i16(array_type(vec![1], INT32))?;
                assert_eq!(a.shape(), &[1]);
                assert_eq!(a[[0]], -123456i32 as i16);
            }
            {
                let a = v.to_ndarray_u32(array_type(vec![1], INT32))?;
                assert_eq!(a.shape(), &[1]);
                assert_eq!(a[[0]], -123456i32 as u32);
            }
            {
                let a = v.to_ndarray_i32(array_type(vec![1], INT32))?;
                assert_eq!(a.shape(), &[1]);
                assert_eq!(a[[0]], -123456i32 as i32);
            }
            {
                let a = v.to_ndarray_u64(array_type(vec![1], INT32))?;
                assert_eq!(a.shape(), &[1]);
                assert_eq!(a[[0]], 4294843840u64);
            }
            {
                let a = v.to_ndarray_i64(array_type(vec![1], INT32))?;
                assert_eq!(a.shape(), &[1]);
                assert_eq!(a[[0]], 4294843840i64);
            }
            Ok(())
        }()
        .unwrap();
    }
}
