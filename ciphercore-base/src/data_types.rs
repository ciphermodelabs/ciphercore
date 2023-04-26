//! Types used within CipherCore and related functions.
use crate::constants::type_size_limit_constants;
use crate::errors::Error;
use crate::errors::Result;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fmt::Display;
use std::fmt::Write;
use std::str::FromStr;
use std::sync::Arc;

#[cfg(feature = "py-binding")]
use pywrapper_macro::{enum_to_struct_wrapper, fn_wrapper, impl_wrapper};

/// A structure that represents a scalar type.
///
/// The supported scalar types is the standard set of scalar types:
/// - `BIT`: a single bit
/// - `U8`: an integer in the range [0, 2<sup>8</sup>]
/// - `I8`: an integer in the range [-2<sup>7</sup>, 2<sup>7</sup> - 1]
/// - `U16`: an integer in the range [0, 2<sup>16</sup>]
/// - `I16`: an integer in the range [-2<sup>15</sup>, 2<sup>15</sup> - 1]
/// - `U32`: an integer in the range [0, 2<sup>32</sup>]
/// - `I32`: an integer in the range [-2<sup>31</sup>, 2<sup>31</sup> - 1]
/// - `U64`: an integer in the range [0, 2<sup>64</sup>]
/// - `I64`: an integer in the range [-2<sup>63</sup>, 2<sup>63</sup> - 1]
///
/// # Examples
///
/// ```
/// # use ciphercore_base::data_types::{BIT, UINT8, UINT64};
/// # use ciphercore_base::data_types::{array_type, vector_type, tuple_type, named_tuple_type};
/// let t0 = BIT;
/// let t1 = array_type(vec![2, 1, 4], t0);
/// let t2 = vector_type(10, t1);
/// let t3 = tuple_type(vec![t2]);
/// assert_eq!("(<bit[2, 1, 4]{10}>)", t3.to_string());
///
/// let t4 = ("Name".to_owned(), array_type(vec![100], UINT8));
/// let t5 = ("Zip".to_owned(), array_type(vec![2], UINT64));
/// let t6 = named_tuple_type(vec![t4, t5]);
/// assert_eq!("(\\\"Name\\\": u8[100], \\\"Zip\\\": u64[2])", t6.to_string());
/// ```
///
#[derive(PartialEq, Eq, Clone, Copy, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[cfg_attr(feature = "py-binding", enum_to_struct_wrapper)]
pub enum ScalarType {
    Bit,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    U128,
    I128,
}

#[cfg_attr(feature = "py-binding", impl_wrapper)]
impl ScalarType {
    /// Deprecated. Use is_signed() instead.
    /// Tests whether this scalar type is signed.
    ///
    /// # Returns
    ///
    /// `true`, if this is a signed scalar, else `false`
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::data_types::{ScalarType, INT16, UINT64};
    /// let t0 = UINT64;
    /// assert!(!t0.get_signed());
    /// let t1 = INT16;
    /// assert!(t1.get_signed());
    /// ```
    pub fn get_signed(&self) -> bool {
        self.is_signed()
    }

    /// Tests whether this scalar type is signed.
    ///
    /// # Returns
    ///
    /// `true`, if this is a signed scalar, else `false`
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::data_types::{ScalarType, INT16, UINT64};
    /// let t0 = UINT64;
    /// assert!(!t0.is_signed());
    /// let t1 = INT16;
    /// assert!(t1.is_signed());
    /// ```
    pub fn is_signed(&self) -> bool {
        match self {
            // Unsigned types.
            ScalarType::Bit => false,
            ScalarType::U8 => false,
            ScalarType::U16 => false,
            ScalarType::U32 => false,
            ScalarType::U64 => false,
            ScalarType::U128 => false,
            // Signed types.
            ScalarType::I8 => true,
            ScalarType::I16 => true,
            ScalarType::I32 => true,
            ScalarType::I64 => true,
            ScalarType::I128 => true,
        }
    }

    /// Returns scalar's modulus value, which defines the range of integers. If it's impossible as for 128-bit types, it returns `None`.
    ///
    /// # Returns
    ///
    /// `modulus` value of the scalar type
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::data_types::{ScalarType, BIT, UINT64};
    /// let two: u128 = 2;
    /// let t0 = ScalarType::Bit;
    /// assert_eq!(Some(two), t0.get_modulus());
    /// let modulus: u128 = two.pow(64);
    /// let t1 = UINT64;
    /// assert_eq!(Some(modulus), t1.get_modulus());
    /// ```
    pub fn get_modulus(&self) -> Option<u128> {
        match self {
            ScalarType::Bit => Some(1u128 << 1),
            ScalarType::U8 => Some(1u128 << 8),
            ScalarType::I8 => Some(1u128 << 8),
            ScalarType::U16 => Some(1u128 << 16),
            ScalarType::I16 => Some(1u128 << 16),
            ScalarType::U32 => Some(1u128 << 32),
            ScalarType::I32 => Some(1u128 << 32),
            ScalarType::U64 => Some(1u128 << 64),
            ScalarType::I64 => Some(1u128 << 64),
            ScalarType::U128 => None,
            ScalarType::I128 => None,
        }
    }

    /// Returns the size of a scalar type in bits.
    ///
    /// # Returns
    ///
    /// Size of a scalar type
    pub fn size_in_bits(&self) -> u64 {
        match self {
            ScalarType::Bit => 1,
            ScalarType::U8 => 8,
            ScalarType::I8 => 8,
            ScalarType::U16 => 16,
            ScalarType::I16 => 16,
            ScalarType::U32 => 32,
            ScalarType::I32 => 32,
            ScalarType::U64 => 64,
            ScalarType::I64 => 64,
            ScalarType::U128 => 128,
            ScalarType::I128 => 128,
        }
    }

    pub(super) fn get_unsigned_counterpart(&self) -> ScalarType {
        match self {
            ScalarType::Bit => ScalarType::Bit,
            ScalarType::U8 => ScalarType::U8,
            ScalarType::I8 => ScalarType::U8,
            ScalarType::U16 => ScalarType::U16,
            ScalarType::I16 => ScalarType::U16,
            ScalarType::U32 => ScalarType::U32,
            ScalarType::I32 => ScalarType::U32,
            ScalarType::U64 => ScalarType::U64,
            ScalarType::I64 => ScalarType::U64,
            ScalarType::U128 => ScalarType::U128,
            ScalarType::I128 => ScalarType::U128,
        }
    }
}

/// Scalar type corresponding to bits 0 or 1.
///
/// BIT scalar type corresponds to either 0 or 1 bit.
///
/// BIT is analogous to [numpy.bool_](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool_) from NumPy.
///
/// # Example
///
/// ```
/// # use ciphercore_base::data_types::{BIT, ScalarType};
/// assert_eq!(BIT, ScalarType::Bit);
/// ```
pub const BIT: ScalarType = ScalarType::Bit;

/// Scalar type corresponding to unsigned 8-bit integers.
///
/// UINT8 corresponds to integers from 0 to 2<sup>8</sup>-1, both inclusive.
///
/// UINT8 is analogous to [numpy.uint8](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.uint8) from NumPy.
///
/// # Example
///
/// ```
/// # use ciphercore_base::data_types::{UINT8, ScalarType};
/// let two: u128 = 2;
/// assert_eq!(UINT8, ScalarType::U8);
/// ```
pub const UINT8: ScalarType = ScalarType::U8;

/// Scalar type corresponding to signed 8-bit integers.
///
/// INT8 corresponds to integers from -2<sup>7</sup> to 2<sup>7</sup>-1, both inclusive.
///
/// INT8 is analogous to [numpy.int8](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.int8) from NumPy.
///
/// # Example
///
/// ```
/// # use ciphercore_base::data_types::{INT8, ScalarType};
/// let two: u128 = 2;
/// assert_eq!(INT8, ScalarType::I8);
/// ```
pub const INT8: ScalarType = ScalarType::I8;

/// Scalar type corresponding to unsigned 16-bit integers.
///
/// UINT16 corresponds to integers from 0 to 2<sup>16</sup>-1, both inclusive.
///
/// UINT16 is analogous to [numpy.uint16](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.uint16) from NumPy.
///
/// # Example
///
/// ```
/// # use ciphercore_base::data_types::{UINT16, ScalarType};
/// let two: u128 = 2;
/// assert_eq!(UINT16, ScalarType::U16);
/// ```
pub const UINT16: ScalarType = ScalarType::U16;

/// Scalar type corresponding to signed 16-bit integers.
///
/// INT16 corresponds to integers from -2<sup>15</sup> to 2<sup>15</sup>-1, both inclusive.
///
/// INT16 is analogous to [numpy.int16](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.int16) from NumPy.
///
/// # Example
///
/// ```
/// # use ciphercore_base::data_types::{INT16, ScalarType};
/// let two: u128 = 2;
/// assert_eq!(INT16, ScalarType::I16);
/// ```
pub const INT16: ScalarType = ScalarType::I16;

/// Scalar type corresponding to unsigned 32-bit integers.
///
/// UINT32 corresponds to integers from 0 to 2<sup>32</sup>-1, both inclusive.
///
/// UINT32 is analogous to [numpy.uint32](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.uint32) from NumPy.
///
/// # Example
///
/// ```
/// # use ciphercore_base::data_types::{UINT32, ScalarType};
/// let two: u128 = 2;
/// assert_eq!(UINT32, ScalarType::U32);
/// ```
pub const UINT32: ScalarType = ScalarType::U32;

/// Scalar type corresponding to signed 32-bit integers.
///
/// INT32 corresponds to integers from -2<sup>31</sup> to 2<sup>31</sup>-1, both inclusive.
///
/// INT32 is analogous to [numpy.int32](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.int32) from NumPy.
///
/// # Example
///
/// ```
/// # use ciphercore_base::data_types::{INT32, ScalarType};
/// let two: u128 = 2;
/// assert_eq!(INT32, ScalarType::I32);
/// ```
pub const INT32: ScalarType = ScalarType::I32;

/// Scalar type corresponding to unsigned 64-bit integers.
///
/// UINT64 corresponds to integers from 0 to 2<sup>64</sup>-1, both inclusive.
///
/// UINT64 is analogous to [numpy.uint64](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.uint64) from NumPy.
///
/// # Example
///
/// ```
/// # use ciphercore_base::data_types::{UINT64, ScalarType};
/// let two: u128 = 2;
/// assert_eq!(UINT64, ScalarType::U64);
/// ```
pub const UINT64: ScalarType = ScalarType::U64;

/// Scalar type corresponding to signed 64-bit integers.
///
/// INT64 corresponds to integers from -2<sup>63</sup> to 2<sup>63</sup>-1, both inclusive.
///
/// INT64 is analogous to [numpy.int64](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.int64) from NumPy.
///
/// # Example
///
/// ```
/// # use ciphercore_base::data_types::{INT64, ScalarType};
/// let two: u128 = 2;
/// assert_eq!(INT64, ScalarType::I64);
/// ```
pub const INT64: ScalarType = ScalarType::I64;

/// Scalar type corresponding to unsigned 128-bit integers.
///
/// UINT128 corresponds to integers from 0 to 2<sup>128</sup>-1, both inclusive.
///
/// # Example
///
/// ```
/// # use ciphercore_base::data_types::{UINT128, ScalarType};
/// let two: u128 = 2;
/// assert_eq!(UINT128, ScalarType::U128);
/// ```
pub const UINT128: ScalarType = ScalarType::U128;

/// Scalar type corresponding to signed 128-bit integers.
///
/// INT128 corresponds to integers from -2<sup>127</sup> to 2<sup>127</sup>-1, both inclusive.
///
/// # Example
///
/// ```
/// # use ciphercore_base::data_types::{INT128, ScalarType};
/// let two: u128 = 2;
/// assert_eq!(INT128, ScalarType::I128);
/// ```
pub const INT128: ScalarType = ScalarType::I128;

/// Vector of dimension lengths for each axis of an array.
///
/// ArrayShape type could be used for array oriented graph operations such as [Sum](crate::graphs::Operation::Sum), [PermuteAxes](crate::graphs::Operation::PermuteAxes), [Get](crate::graphs::Operation::Get), [Stack](crate::graphs::Operation::Get) etc.
///
/// ArrayShape is valid if
///
/// * vector is not empty,
/// * all the dimension lengths are non-zero,
/// * total array capacity (product of dimension lengths) is less than [u64::MAX].
///
/// ArrayShape is analogous to [numpy.ndarray.shape](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html).
///
/// # Example
///
/// ```
/// # use ciphercore_base::data_types::{UINT32, ArrayShape, array_type};
/// let a_shape0: ArrayShape = vec![10, 1, 20];
/// let at0 = array_type(a_shape0, UINT32);
/// assert!(at0.is_valid());
/// let a_shape1: ArrayShape = vec![];
/// let at1 = array_type(a_shape1, UINT32);
/// assert!(!at1.is_valid());
/// let a_shape2: ArrayShape = vec![10, 0];
/// let at2 = array_type(a_shape2, UINT32);
/// assert!(!at2.is_valid());
/// let a_shape3: ArrayShape = vec![(u64::MAX/2), 3];
/// let at3 = array_type(a_shape3, UINT32);
/// assert!(!at3.is_valid());
/// ```
pub type ArrayShape = Vec<u64>;

/// This enum represents the input or output type of a computation [Node](crate::graphs::Node) within the parent computational [Graph](crate::graphs::Graph).
#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize, Hash)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[cfg_attr(feature = "py-binding", enum_to_struct_wrapper)]
pub enum Type {
    /// Each scalar corresponds to a signed or an unsigned number modulo `m`, where `m` = {2, 2<sup>8</sup>, 2<sup>16</sup>, 2<sup>32</sup>, 2<sup>64</sup>}.
    ///
    /// Scalar types supported by CipherCore are provided as parameters. These scalar types could be [BIT], [UINT8], [INT8], [UINT16], [INT16], [UINT32], [INT32], [UINT64] or [INT64].
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::{create_context};
    /// # use ciphercore_base::data_types::{BIT, scalar_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let t = scalar_type(BIT);
    /// let i0 = g.input(t.clone()).unwrap();
    /// assert_eq!(i0.get_type().unwrap(), t);
    /// ```
    Scalar(ScalarType),

    /// Array entries are scalars and arrays could be multi-dimensional.
    ///
    /// Array is analogous to NumPy's [numpy.array](https://numpy.org/doc/stable/reference/generated/numpy.array.html).
    ///
    /// An array is defined using a valid shape and a supported scalar type.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::{create_context};
    /// # use ciphercore_base::data_types::{UINT32, array_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let a_shape = vec![10, 1, 20];
    /// let at = array_type(a_shape, UINT32);
    /// assert!(at.is_valid());
    /// let i0 = g.input(at.clone()).unwrap();
    /// assert_eq!(i0.get_type().unwrap(), at);
    /// ```
    Array(ArrayShape, ScalarType),

    /// Vector type is a one-dimensional list of entries of some constituent type,
    /// which can be any valid type including the vector or tuple type.
    ///
    /// A vector type is defined by the number of entries and a pointer to the
    /// type of those entries.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::graphs::{create_context};
    /// # use ciphercore_base::data_types::{UINT32, array_type, vector_type};
    /// let c = create_context().unwrap();
    /// let g = c.create_graph().unwrap();
    /// let a_shape = vec![10, 1, 20];
    /// let at = array_type(a_shape, UINT32);
    /// let vt0 = vector_type(10, at);
    /// assert!(vt0.is_valid());
    /// let vt1 = vector_type(10, vt0);
    /// let i0 = g.input(vt1).unwrap();
    /// assert!(i0.get_type().unwrap().is_valid());
    /// ```
    Vector(u64, TypePointer),

    /// Tuples are fixed-length lists consisting of entries of given type(s).
    ///
    /// tuple(`T`<sub>0</sub>, `T`<sub>1</sub>, ..., `T`<sub>n-1</sub>) corresponds to a list of types `T`<sub>0</sub>, `T`<sub>1</sub>, ..., `T`<sub>n-1</sub>.
    ///
    /// tuple() can be interpreted as a void type.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::data_types::{UINT32, array_type, vector_type, tuple_type};
    /// let a_shape = vec![10, 1, 20];
    /// let at = array_type(a_shape, UINT32);
    /// let vt0 = vector_type(10, at.clone());
    /// let vt1 = vector_type(10, at);
    /// let tt = tuple_type(vec![vt0, vt1]);
    /// assert!(tt.is_valid());
    /// ```
    Tuple(Vec<TypePointer>),

    /// Named tuples are fixed-length lists that consist of tuples of element-name and associated element-type.
    ///
    /// An element's name can be used to access a particular element within the named tuple.
    ///
    /// A valid named tuple should have all the element-types valid.
    ///
    /// Element-names for the element-types should be unique.
    ///
    /// Named tuple is similar to the tuple type.
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::data_types::{UINT8, UINT64, scalar_type, array_type, named_tuple_type};
    /// let st = scalar_type(UINT64);
    /// let at = array_type(vec![64], UINT8);
    /// let nt = named_tuple_type(vec![("ID".to_owned(), st), ("Name".to_owned(), at)]);
    /// assert!(nt.is_valid());
    /// ```
    NamedTuple(Vec<(String, TypePointer)>),
}

/// Pointer to a type.
pub(super) type TypePointer = Arc<Type>;

/// Void type instance for type.
///
/// `VOID_TYPE` is a type of an empty tuple.
///
/// # Example
///
/// ```
/// # use ciphercore_base::data_types::{VOID_TYPE, tuple_type};
/// assert_eq!(VOID_TYPE.clone(), tuple_type(vec![]));
/// ```
pub const VOID_TYPE: Type = Type::Tuple(vec![]);

/// Tests whether a given array shape is valid.
///
/// Array shape `s` is invalid if `s` is empty, or one of the dimension sizes is 0 or
/// if the total element capacity of the array is greater than [u64::MAX].
///
/// # Arguments
///
/// `s` - shape of the array to be tested
///
/// # Returns
///
/// `true`, if the input array shape `s` is valid, otherwise `false`
pub fn is_valid_shape(s: ArrayShape) -> bool {
    if s.is_empty() {
        return false;
    }
    for x in &s {
        if *x == 0 {
            return false;
        }
    }
    let mut tmp = u64::MAX;
    for x in &s {
        tmp /= *x;
    }
    tmp != 0
}

#[cfg_attr(feature = "py-binding", impl_wrapper)]
impl Type {
    /// Tests whether a given type is valid.
    ///
    /// Recursively checks if all the arrays that constitute the type have valid shapes, and all the named tuples have unique field names.
    ///
    /// # Returns
    ///
    /// `true`, if type instance and its sub-type(s) are valid, otherwise `false`
    ///
    /// # Example
    /// ```
    /// # use ciphercore_base::data_types::{array_type, ScalarType, Type, INT16};
    /// let s0 = INT16;
    /// let t0 = Type::Scalar(s0.clone());
    /// assert!(t0.is_valid());
    ///
    /// let a1 = array_type(vec![2, 0, 2], s0.clone());
    /// assert!(!a1.is_valid());
    ///
    /// let a2 = array_type(vec![2, 1, 2], s0);
    /// assert!(a2.is_valid());
    /// ```
    pub fn is_valid(&self) -> bool {
        match self {
            Type::Scalar(_) => true,
            Type::Array(shape, _) => is_valid_shape(shape.clone()),
            Type::Vector(_, element_type) => element_type.is_valid(),
            Type::Tuple(element_types) => element_types.iter().all(|t| t.is_valid()),
            Type::NamedTuple(elements) => {
                let mut names: Vec<String> = elements.iter().map(|(x, _)| x.clone()).collect();
                names.sort();
                names.dedup();
                let names_valid = names.len() == elements.len();
                let types_valid = elements.iter().all(|(_, y)| y.is_valid());
                names_valid && types_valid
            }
        }
    }

    /// Tests if a type is scalar.
    ///
    /// # Returns
    ///
    /// `true`, if the type instance is scalar, otherwise `false`
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::data_types::{ScalarType, Type, array_type, INT16};
    /// let t0 = Type::Scalar(INT16);
    /// assert!(t0.is_scalar());
    /// let a1 = array_type(vec![2, 1, 2], INT16);
    /// assert!(!a1.is_scalar());
    /// ```
    pub fn is_scalar(&self) -> bool {
        matches!(self, Type::Scalar(_))
    }

    /// Tests if a type is array.
    ///
    /// # Returns
    ///
    /// `true`, if type is an array, otherwise `false`
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::data_types::{array_type, ScalarType, Type, INT16};
    /// let t0 = Type::Scalar(INT16);
    /// assert!(!t0.is_array());
    ///
    /// let a1 = array_type(vec![2, 1, 2], INT16);
    /// assert!(a1.is_array());
    /// ```
    pub fn is_array(&self) -> bool {
        matches!(self, Type::Array(_, _))
    }

    /// Tests if a type is vector.
    ///
    /// # Returns
    ///
    /// `true`, if type is a vector, otherwise `false`
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::data_types::{array_type, vector_type, Type, ScalarType, INT16};
    /// let t0 = Type::Scalar(INT16);
    /// assert!(!t0.is_vector());
    /// let a0 = array_type(vec![2, 1, 2], INT16);
    /// assert!(!a0.is_vector());
    /// let v0 = vector_type(2, a0);
    /// assert!(v0.is_vector());
    /// ```
    pub fn is_vector(&self) -> bool {
        matches!(self, Type::Vector(_, _))
    }

    /// Tests if a type is tuple.
    ///
    /// # Returns
    ///
    /// `true`, if type is a tuple, otherwise `false`
    ///
    /// # Example
    /// ```
    /// # use ciphercore_base::data_types::{tuple_type, array_type, vector_type, Type, ScalarType, INT32};
    /// let s0 = INT32;
    /// let t0 = Type::Scalar(s0.clone());
    /// assert!(!t0.is_tuple());
    /// let a0 = array_type(vec![2, 1, 2], s0.clone());
    /// assert!(!a0.is_tuple());
    /// let v0 = vector_type(2, a0.clone());
    /// assert!(!v0.is_tuple());
    /// let v1 = vector_type(2, a0);
    /// let t1 = tuple_type(vec![v0, v1]);
    /// assert!(t1.is_tuple());
    /// ```
    pub fn is_tuple(&self) -> bool {
        matches!(self, Type::Tuple(_))
    }

    /// Tests if a type is named tuple.
    ///
    /// # Returns
    ///
    /// `true`, if type is a named tuple, otherwise `false`
    ///
    /// # Example
    /// ```
    /// # use ciphercore_base::data_types::{named_tuple_type, array_type, ScalarType, INT16};
    /// let s0 = INT16;
    /// let a0 = array_type(vec![2, 1, 2], s0.clone());
    /// assert!(!a0.is_tuple());
    /// let nt0 = named_tuple_type(vec![
    ///     ("Name".to_owned(), a0.clone()),
    ///     ("Zip".to_owned(), a0)
    /// ]);
    /// assert!(nt0.is_named_tuple());
    /// ```
    pub fn is_named_tuple(&self) -> bool {
        matches!(self, Type::NamedTuple(_))
    }

    /// Returns the underlying scalar type of a scalar or an array.
    ///
    /// # Panics
    ///
    /// Panics if `self` is neither a scalar nor an array
    ///
    /// # Returns
    ///
    /// Copy of the underlying scalar type for a scalar or an array
    ///
    /// # Example
    /// ```
    /// # use ciphercore_base::data_types::{ScalarType, Type, array_type, INT16};
    /// let s0 = INT16;
    /// let st0 = Type::Scalar(s0.clone());
    /// let a0 = array_type(vec![2, 1, 2], s0.clone());
    /// assert!(st0.get_scalar_type().eq(&s0));
    /// assert!(a0.get_scalar_type().eq(&s0));
    /// ```
    pub fn get_scalar_type(&self) -> ScalarType {
        if let Type::Scalar(st) | Type::Array(_, st) = self {
            *st
        } else {
            panic!("Can't get scalar type");
        }
    }

    /// Returns the shape of an array.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not an array
    ///
    /// # Returns
    ///
    /// Copy of the shape of an array
    ///
    /// # Example
    /// ```
    /// # use ciphercore_base::data_types::{ScalarType, Type, array_type, INT16};
    /// let s0 = INT16;
    /// let st0 = Type::Scalar(s0.clone());
    /// let a0_shape = vec![2, 1, 2];
    /// let a0 = array_type(a0_shape.clone(), s0.clone());
    /// assert_eq!(a0_shape, a0.get_shape());
    /// ```
    pub fn get_shape(&self) -> ArrayShape {
        if let Type::Array(shape, _) = self {
            shape.clone()
        } else {
            panic!("Can't get shape of {self:?})");
        }
    }

    /// Returns the array shape for an array type
    /// or \[1\] for a scalar type.
    ///
    /// # Panics
    ///
    /// Panics if `self` is neither a scalar nor an array
    ///
    /// # Returns
    ///
    /// * Copy of the shape of a given array, or
    /// * Shape \[1\], if a scalar type is given
    ///
    /// # Example
    ///
    /// ```
    /// # use ciphercore_base::data_types::{ScalarType, Type, array_type, INT16};
    /// let s0 = INT16;
    /// let st0 = Type::Scalar(s0.clone());
    /// let a0_shape = vec![2, 1, 2];
    /// let a0 = array_type(a0_shape.clone(), s0.clone());
    /// assert!(a0.get_dimensions().eq(&a0_shape));
    /// assert_eq!(vec![1], st0.get_dimensions());
    /// ```
    pub fn get_dimensions(&self) -> ArrayShape {
        if let Type::Array(shape, _) = self {
            shape.clone()
        } else if let Type::Scalar(_) = self {
            vec![1]
        } else {
            panic!("Can't get shape");
        }
    }

    /// Returns the size of a type in bits.
    ///
    /// Returns a runtime error if type is not valid.
    ///
    /// Within named tuple types, the bit size of name strings is omitted.
    ///
    /// # Returns
    ///
    /// Type size in bits
    pub fn size_in_bits(&self) -> Result<u64> {
        get_size_in_bits(self.clone())
    }

    /// Serializes type to string.
    ///
    /// # Returns
    ///
    /// Json string generated from the given type
    #[cfg_attr(not(feature = "py-binding"), allow(dead_code))]
    fn to_json_string(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    /// Deserializes type from string.
    ///
    /// # Arguments
    ///
    /// `s` - json string
    ///
    /// # Returns
    ///
    /// New type constructed from the given json
    #[cfg_attr(not(feature = "py-binding"), allow(dead_code))]
    fn from_json_string(s: String) -> Result<Type> {
        Ok(serde_json::from_str::<Type>(s.as_str())?)
    }
}

impl Type {
    /// Returns a vector of tuples created from the named tuple.
    ///
    /// This function reverse the result of `named_tuple_type()`.
    ///
    /// # Returns
    ///
    /// Vector of tuples (type name, type)
    ///
    /// # Example
    /// ```
    /// # use ciphercore_base::data_types::{UINT8, UINT32, Type, array_type, vector_type, named_tuple_type};
    /// let st0 = Type::Scalar(UINT8);
    /// let st1 = Type::Scalar(UINT32);
    /// let vec_t = vec![("age".to_owned(), st0), ("zip".to_owned(), st1)];
    /// let t = named_tuple_type(vec_t.clone());
    /// assert_eq!(vec_t, t.get_named_types().unwrap());
    /// ```
    pub fn get_named_types(&self) -> Result<Vec<(String, Type)>> {
        // TODO: pyo3 support
        if let Type::NamedTuple(v) = self {
            let mut res = vec![];
            for (name, t) in v {
                res.push((name.clone(), (*t.to_owned()).clone()));
            }
            Ok(res)
        } else {
            Err(runtime_error!(
                "Can't get named types. Input type must be NamedTuple."
            ))
        }
    }

    #[doc(hidden)]
    pub fn get_names(&self) -> Result<Vec<String>> {
        if let Type::NamedTuple(strings_types) = self.clone() {
            let v = strings_types.iter().map(|st| st.0.clone()).collect();
            Ok(v)
        } else {
            Err(runtime_error!(
                "Can't get named types. Input type must be NamedTuple."
            ))
        }
    }
}

// column header -> column type
#[doc(hidden)]
pub type HeadersTypes = Vec<(String, Arc<Type>)>;

#[doc(hidden)]
pub fn get_named_types(t: &Type) -> Result<&HeadersTypes> {
    if let Type::NamedTuple(v) = t {
        Ok(v)
    } else {
        Err(runtime_error!(
            "Can't get named types. Input type must be NamedTuple."
        ))
    }
}

/// Returns a new type for scalars created from a given scalar type.
///
/// This is a helper function to create a `Type::Scalar(_)`.
///
/// `st` should be a supported scalar type from one of [BIT], [INT8], [INT16],
/// [INT32], [INT64], [UINT8], [UINT16], [UINT32], and [UINT64].
///
/// # Arguments
///
/// `st` - valid and supported scalar type
///
/// # Returns
///
/// New type from the given scalar type
///
/// # Example
/// ```
/// # use ciphercore_base::data_types::{INT32, scalar_type, Type};
/// let s = INT32;
/// let st = scalar_type(s.clone());
/// assert!(st.is_scalar());
/// assert!(s.eq(&st.get_scalar_type()));
/// ```
#[cfg_attr(feature = "py-binding", fn_wrapper)]
pub fn scalar_type(st: ScalarType) -> Type {
    Type::Scalar(st)
}

/// Returns a new array type with a given shape and scalar type.
///
/// This is a helper function to create a `Type::Array(_, _)`.
///
/// `shape` of the array should be valid. `shape` is invalid if `s` is empty, or one of the dimension sizes is 0 or
/// if the total element capacity of the array is greater than [u64::MAX].
///
/// `st` should be a supported scalar from one of [BIT], [INT8], [INT16],
/// [INT32], [INT64], [UINT8], [UINT16], [UINT32], and [UINT64].
///
/// # Arguments
///
/// * `shape` - valid shape of array
/// * `st` - valid and supported scalar type
///
/// # Returns
///
/// New array type from the given shape and scalar type
///
/// # Example
/// ```
/// # use ciphercore_base::data_types::{INT32, scalar_type, Type, array_type};
/// let s0 = INT32;
/// let a0_shape = vec![2, 1, 2];
/// let a0 = array_type(a0_shape.clone(), s0.clone());
/// assert!(a0.is_array());
/// assert!(a0_shape.eq(&a0.get_shape()));
/// assert!(s0.eq(&a0.get_scalar_type()));
/// ```
#[cfg_attr(feature = "py-binding", fn_wrapper)]
pub fn array_type(shape: ArrayShape, st: ScalarType) -> Type {
    Type::Array(shape, st)
}

/// Returns a new vector type with a given length and underlying type.
///
/// This is a helper function to create a `Type::Vector(_, _)`.
///
/// # Arguments
///
/// * `n` - required length of the vector
/// * `t` - valid component-type of the vector
///
/// # Returns
///
/// New vector type with `n` instances of the given type
///
/// # Example
/// ```
/// # use ciphercore_base::data_types::{INT32, Type, array_type, vector_type};
/// let s0 = INT32;
/// let a0_shape = vec![2, 1, 2];
/// let t = array_type(a0_shape.clone(), s0.clone());
/// let n = 5;
/// let v0 = vector_type(n, t.clone());
/// let (n1, t1) = match v0 {
///     Type::Vector(n0, t0) => (Some(n0), Some(t0)),
///     _ => (None, None)
/// };
/// assert!(!n1.is_none());
/// assert!(!t1.is_none());
/// assert!(n == n1.unwrap());
/// assert!(t.eq(&t1.unwrap()));
/// ```
#[cfg_attr(feature = "py-binding", fn_wrapper)]
pub fn vector_type(n: u64, t: Type) -> Type {
    Type::Vector(n, Arc::new(t))
}

/// Returns a new tuple type with underlying types provided by a given vector.
///
/// This is a helper function to create a `Type::Tuple(_)`.
///
/// All the types in the given vector need not be the same.
///
/// # Arguments
///
/// `v` - vector of required types
///
/// # Returns
///
/// New tuple type created from the given vector of types
///
/// # Example
/// ```
/// # use ciphercore_base::data_types::{INT32, Type, array_type, vector_type, tuple_type};
/// let st = INT32;
/// let s = Type::Scalar(st);
/// let a_shape = vec![2, 1, 2];
/// let a = array_type(a_shape.clone(), st);
/// let n = 5;
/// let v = vector_type(n, a.clone());
/// let vec_types = vec![s.clone(), a.clone(), v.clone()];
/// let t = tuple_type(vec_types.clone());
/// assert!(t.is_tuple());
/// ```
#[cfg_attr(feature = "py-binding", fn_wrapper)]
pub fn tuple_type(v: Vec<Type>) -> Type {
    let mut vp = vec![];
    for t in v {
        vp.push(Arc::new(t.clone()));
    }
    Type::Tuple(vp)
}

/// Returns a new named tuple type with underlying names and types provided by a given vector.
///
/// This is a helper function to create a `Type::NamedTuple(_)`.
///
/// All the types in the given vector need not be the same.
///
/// # Arguments
///
/// `v` - vector of tuples (type name, type)
///
/// # Returns
///
/// New named tuple type created from the given vector of names and types
///
/// # Example
/// ```
/// # use ciphercore_base::data_types::{UINT8, UINT32, Type, array_type, vector_type, named_tuple_type};
/// let st0 = Type::Scalar(UINT8);
/// let st1 = Type::Scalar(UINT32);
/// let vec_t = vec![("age".to_owned(), st0), ("zip".to_owned(), st1)];
/// let t = named_tuple_type(vec_t);
/// assert!(t.is_named_tuple());
/// ```
#[cfg_attr(feature = "py-binding", fn_wrapper)]
pub fn named_tuple_type(v: Vec<(String, Type)>) -> Type {
    let mut vp = vec![];
    for (s, t) in v {
        vp.push((s.clone(), Arc::new(t.clone())));
    }
    Type::NamedTuple(vp)
}

fn form_array_shape_str(array_shape: ArrayShape) -> String {
    let mut array_shape_str = String::from("");
    array_shape_str.push('[');
    let mut dim_len_iter = array_shape.iter();
    if let Some(&d) = dim_len_iter.next() {
        write!(array_shape_str, "{d}").unwrap();
    }
    for dimension_length in dim_len_iter {
        write!(array_shape_str, ", {dimension_length}").unwrap();
    }
    array_shape_str.push(']');
    array_shape_str
}

fn form_array_type_str(shape: ArrayShape, scalar_type: ScalarType) -> String {
    format!("{}{}", scalar_type, form_array_shape_str(shape))
}

fn form_vector_type_str(number_of_components: u64, type_pointer: TypePointer) -> String {
    format!("<{}{{{}}}>", *type_pointer, number_of_components)
}

fn form_tuple_vec_type_str(vec_type_pointer: Vec<TypePointer>) -> String {
    let mut vec_type_str = String::from("");
    let mut vec_type_pointer_iter = vec_type_pointer.iter();
    if let Some(type_pointer) = vec_type_pointer_iter.next() {
        vec_type_str.push_str(&((**type_pointer).clone().to_string()));
    }
    for type_pointer in vec_type_pointer_iter {
        write!(vec_type_str, ", {}", (**type_pointer).clone()).unwrap();
    }
    vec_type_str
}

fn form_named_tuple_vec_type_str(named_tup_vec_type_pointer: Vec<(String, TypePointer)>) -> String {
    let mut named_tup_vec_type_str = String::from("");
    let mut named_tup_vec_type_pointer_iter = named_tup_vec_type_pointer.iter();

    if let Some((name_str, type_pointer)) = named_tup_vec_type_pointer_iter.next() {
        let named_tuple_type =
            format!("\\\"{}\\\": {}", name_str.clone(), (**type_pointer).clone());
        named_tup_vec_type_str.push_str(&named_tuple_type);
    }
    for named_tup_vec in named_tup_vec_type_pointer_iter {
        let (name_str, type_pointer) = named_tup_vec;
        let named_tuple_type = format!(", \\\"{}\\\": {}", name_str, (**type_pointer).clone());
        named_tup_vec_type_str.push_str(&named_tuple_type);
    }
    named_tup_vec_type_str
}

impl fmt::Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let scalar_type = self;
        let mut scalar_type_string = String::from("");
        let bit_size = scalar_size_in_bits(*scalar_type);
        if bit_size == 1 {
            write!(scalar_type_string, "bit").unwrap();
        } else {
            if scalar_type.is_signed() {
                scalar_type_string.push('i');
            } else {
                scalar_type_string.push('u');
            }
            write!(scalar_type_string, "{bit_size}").unwrap();
        }
        write!(f, "{scalar_type_string}")
    }
}
impl fmt::Debug for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl FromStr for ScalarType {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "bit" => Ok(BIT),
            "u8" => Ok(UINT8),
            "i8" => Ok(INT8),
            "u16" => Ok(UINT16),
            "i16" => Ok(INT16),
            "u32" => Ok(UINT32),
            "i32" => Ok(INT32),
            "u64" => Ok(UINT64),
            "i64" => Ok(INT64),
            "u128" => Ok(UINT128),
            "i128" => Ok(INT128),
            _ => Err(runtime_error!(
                "Unknown scalar type. Expected b|u8|i8|u16|i16|u32|i32|u64|i64|u128|i128."
            )),
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let type_string = match self {
            Type::Scalar(scalar_type) => {
                format!("{scalar_type}")
            }
            Type::Array(shape, scalar_type) => form_array_type_str(shape.clone(), *scalar_type),
            Type::Vector(number_of_components, element_type) => {
                form_vector_type_str(*number_of_components, element_type.clone())
            }
            Type::Tuple(element_types) => {
                format!("({})", form_tuple_vec_type_str(element_types.clone()))
            }
            Type::NamedTuple(elements) => {
                format!("({})", form_named_tuple_vec_type_str(elements.clone()))
            }
        };
        write!(f, "{type_string}")
    }
}

/// Returns the size of a given scalar type in bits.
///
/// # Arguments
///
/// `t` - scalar type
///
/// # Returns
///
/// Size of a scalar type
///
/// # Example
///
/// ```
/// # use ciphercore_base::data_types::{scalar_size_in_bits, BIT, UINT8, INT64, UINT64};
/// assert_eq!(scalar_size_in_bits(BIT), 1);
/// assert_eq!(scalar_size_in_bits(UINT8), 8);
/// assert_eq!(scalar_size_in_bits(INT64), 64);
/// assert_eq!(scalar_size_in_bits(UINT64), 64);
/// ```
pub fn scalar_size_in_bits(t: ScalarType) -> u64 {
    t.size_in_bits()
}

pub(crate) fn scalar_size_in_bytes(t: ScalarType) -> u64 {
    (scalar_size_in_bits(t) + 7) / 8
}

/// Returns the size of a given type in bits.
///
/// Returns a runtime error if `t` is not valid.
///
/// Within named tuple types, the bit size of name strings is omitted.
///
/// # Arguments
///
/// `t` - type
///
/// # Returns
///
/// Type size in bits
///
/// # Example
///
/// ```
/// # use ciphercore_base::data_types::{Type, UINT64, get_size_in_bits, scalar_type, array_type, vector_type, tuple_type, named_tuple_type};
/// assert_eq!(get_size_in_bits(Type::Scalar(UINT64)).unwrap(), 64);
/// assert_eq!(get_size_in_bits(array_type(vec![10], UINT64)).unwrap(), 640);
/// assert_eq!(get_size_in_bits(vector_type(2, array_type(vec![2, 2], UINT64))).unwrap(), 512);
/// assert_eq!(get_size_in_bits(tuple_type(vec![Type::Scalar(UINT64), array_type(vec![2], UINT64)])).unwrap(), 192);
/// assert_eq!(get_size_in_bits(named_tuple_type(vec![("ID".to_owned(), scalar_type(UINT64)), ("Worth".to_owned(), scalar_type(UINT64))])).unwrap(), 128);
/// ```
pub fn get_size_in_bits(t: Type) -> Result<u64> {
    if !t.is_valid() {
        return Err(runtime_error!("Invalid type!"));
    }
    let result = match t {
        Type::Scalar(st) => scalar_size_in_bits(st),
        Type::Array(s, st) => {
            let sz = scalar_size_in_bits(st);
            let mut pr: u64 = 1;
            for x in &s {
                pr = pr
                    .checked_mul(*x)
                    .ok_or_else(|| runtime_error!("multiply overflow!"))?;
            }
            sz.checked_mul(pr)
                .ok_or_else(|| runtime_error!("multiply overflow!"))?
        }
        Type::Vector(length, element_type) => {
            let et = get_size_in_bits((*element_type).clone())?;
            length
                .checked_mul(et)
                .ok_or_else(|| runtime_error!("multiply overflow!"))?
        }
        Type::Tuple(types) => {
            let mut total_size: u64 = 0;
            for element_type in types {
                let et = get_size_in_bits((*element_type).clone())?;
                total_size = total_size
                    .checked_add(et)
                    .ok_or_else(|| runtime_error!("add overflow!"))?;
            }
            total_size
        }
        Type::NamedTuple(names_types) => {
            let mut total_size: u64 = 0;
            for (_, element_type) in names_types {
                let et = get_size_in_bits((*element_type).clone())?;
                total_size = total_size
                    .checked_add(et)
                    .ok_or_else(|| runtime_error!("add overflow!"))?;
            }
            total_size
        }
    };
    Ok(result)
}

/// Returns the size of the storage for `t` in bits.
/// We use this function to have an estimation for how much memory nodes with each type consume
/// For getting the actual size of types you need to use get_size_in_bits
pub(super) fn get_size_estimation_in_bits(t: Type) -> Result<u64> {
    if !t.is_valid() {
        return Err(runtime_error!("Invalid type!"));
    }
    let result = match t {
        Type::Scalar(st) => scalar_size_in_bits(st),
        Type::Array(s, st) => {
            let sz = scalar_size_in_bits(st);
            let mut pr: u64 = 1;
            for x in &s {
                pr = pr
                    .checked_mul(*x)
                    .ok_or_else(|| runtime_error!("multiply overflow!"))?;
            }
            let pr_plus_one = pr
                .checked_add(1)
                .ok_or_else(|| runtime_error!("add overflow!"))?;
            sz.checked_mul(pr_plus_one)
                .ok_or_else(|| runtime_error!("multiply overflow!"))?
        }
        Type::Vector(length, element_type) => {
            let et = get_size_estimation_in_bits((*element_type).clone())?;

            let length_plus_one = length
                .checked_add(1)
                .ok_or_else(|| runtime_error!("add overflow!"))?;
            length_plus_one
                .checked_mul(et)
                .ok_or_else(|| runtime_error!("multiply overflow!"))?
        }
        Type::Tuple(types) => {
            let mut total_size: u64 = 0;
            for element_type in types {
                let et = get_size_estimation_in_bits((*element_type).clone())?;
                total_size = total_size
                    .checked_add(et)
                    .ok_or_else(|| runtime_error!("add overflow!"))?;
            }
            total_size
        }
        Type::NamedTuple(names_types) => {
            let mut total_size: u64 = 0;
            for (_, element_type) in names_types {
                let et = get_size_estimation_in_bits((*element_type).clone())?;
                total_size = total_size
                    .checked_add(et)
                    .ok_or_else(|| runtime_error!("add overflow!"))?;
            }
            total_size
        }
    };
    result
        .checked_add(type_size_limit_constants::TYPE_MEMORY_OVERHEAD)
        .ok_or_else(|| runtime_error!("add overflow!"))
}

/// Returns a vector of types contained in a given type.
///
/// Types supported are vector, tuple and named tuple.
///
/// Number of types contained in the given type should be less than or equal to
/// `TYPES_VECTOR_LENGTH_LIMIT`.
///
/// This function is provided for internal requirements.
///
/// # Arguments
///
/// `t` - type
///
/// # Returns
///
/// Vector of types
///
/// # Example
///
/// ```
/// # use ciphercore_base::data_types::{Type, UINT64, get_types_vector, scalar_type, array_type, vector_type, tuple_type, named_tuple_type};
/// let v = get_types_vector(vector_type(4, scalar_type(UINT64))).unwrap();
/// let v_v = vec![scalar_type(UINT64); 4];
/// assert_eq!(v.len(), v_v.len());
/// for (x, y) in v.iter().zip(v_v.iter()) {
///     assert_eq!(*x.clone(), *y);
/// }
///
/// let types_vec = vec![array_type(vec![2, 2], UINT64), vector_type(2, array_type(vec![3, 3], UINT64))];
/// let tuple_types = get_types_vector(tuple_type(types_vec.clone())).unwrap();
/// assert_eq!(types_vec.len(), tuple_types.len());
/// for (x, y) in tuple_types.iter().zip(types_vec.iter()) {
///     assert_eq!(*x.clone(), *y);
/// }
///
/// let sc_type = scalar_type(UINT64);
/// let n_tuples_vec = vec![("ID".to_owned(), sc_type.clone()), ("Worth".to_owned(), sc_type.clone())];
/// let n_tuples_type_vec = vec![sc_type.clone(); 2];
/// let n_tuple_type = get_types_vector(named_tuple_type(n_tuples_vec.clone())).unwrap();
/// assert_eq!(n_tuples_vec.len(), n_tuple_type.len());
/// for (x, y) in n_tuple_type.iter().zip(n_tuples_type_vec.iter()) {
///     assert_eq!(*x.clone(), *y);
/// }
/// ```
pub fn get_types_vector(t: Type) -> Result<Vec<TypePointer>> {
    match t {
        Type::Vector(length, element_type) => {
            if length > type_size_limit_constants::TYPES_VECTOR_LENGTH_LIMIT as u64 {
                return Err(runtime_error!(
                    "Vector length is greater than TYPES_VECTOR_LENGTH_LIMIT!"
                ));
            }
            let mut result = vec![];
            for _ in 0..length {
                result.push(element_type.clone());
            }
            Ok(result)
        }
        Type::Tuple(types) => {
            let length = types.len();
            if length > type_size_limit_constants::TYPES_VECTOR_LENGTH_LIMIT {
                return Err(runtime_error!(
                    "Tuple length is greater than TYPES_VECTOR_LENGTH_LIMIT!"
                ));
            }
            let mut result = vec![];
            for t in &types {
                result.push(t.clone());
            }
            Ok(result)
        }
        Type::NamedTuple(names_types) => {
            let length = names_types.len();
            if length > type_size_limit_constants::TYPES_VECTOR_LENGTH_LIMIT {
                return Err(runtime_error!(
                    "NamedTuple length is greater than TYPES_VECTOR_LENGTH_LIMIT!"
                ));
            }
            let mut result = vec![];
            for (_, t) in &names_types {
                result.push(t.clone());
            }
            Ok(result)
        }
        _ => Err(runtime_error!("Not a vector type!")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparison() {
        assert_eq!(INT32, INT32);
        assert_eq!(scalar_type(INT8), scalar_type(INT8));
        assert_eq!(array_type(vec![10, 20], BIT), array_type(vec![10, 20], BIT));
        assert!(INT64 != UINT64);
        assert!(scalar_type(INT8) != scalar_type(INT64));
        assert!(array_type(vec![10, 20], BIT) != array_type(vec![10, 20], INT32));
        assert!(array_type(vec![10], BIT) != array_type(vec![10, 20], BIT));
        assert!(array_type(vec![20, 10], BIT) != array_type(vec![10, 20], BIT));
        assert!(array_type(vec![10], BIT) != vector_type(10, scalar_type(BIT)));
        assert!(tuple_type(vec![array_type(vec![100], BIT)]) != tuple_type(vec![]));
        assert!(
            named_tuple_type(vec![("Name".to_owned(), array_type(vec![100], BIT))])
                != named_tuple_type(vec![("Gender".to_owned(), array_type(vec![100], BIT))])
        );
    }

    #[test]
    fn test_debug() {
        let t = array_type(vec![10, 10], UINT32);
        assert_eq!(format!("{:?}", t), "Array([10, 10], u32)");
        let t2 = named_tuple_type(vec![("Name".to_owned(), array_type(vec![100], BIT))]);
        assert_eq!(
            format!("{:?}", t2),
            "NamedTuple([(\"Name\", Array([100], bit))])"
        );
    }

    #[test]
    fn test_clone() {
        let t = array_type(vec![10, 10], UINT32);
        let t1 = t.clone();
        assert_eq!(format!("{:?}", t), "Array([10, 10], u32)");
        assert_eq!(format!("{:?}", t1), "Array([10, 10], u32)");
        let t2 = named_tuple_type(vec![("Name".to_owned(), array_type(vec![100], BIT))]);
        let t3 = t2.clone();
        assert_eq!(
            format!("{:?}", t3),
            "NamedTuple([(\"Name\", Array([100], bit))])"
        );
    }

    #[test]
    fn test_serialization() {
        let t = INT64;
        let se = serde_json::to_string(&t).unwrap();
        assert_eq!(se, "\"i64\"");
        let de: ScalarType = serde_json::from_str(&se).unwrap();
        assert_eq!(t, de);
        assert!(serde_json::from_str::<ScalarType>("{{{{{{{{{{{{{").is_err());
        assert!(serde_json::from_str::<ScalarType>("\"randomstring\"").is_err());
        let t = array_type(vec![10, 10], UINT32);
        let se = serde_json::to_string(&t).unwrap();
        assert_eq!(se, "{\"Array\":[[10,10],\"u32\"]}");
        let de: Type = serde_json::from_str(&se).unwrap();
        assert_eq!(t, de);
        let t = tuple_type(vec![scalar_type(BIT), tuple_type(vec![scalar_type(BIT)])]);
        let se = serde_json::to_string(&t).unwrap();
        assert_eq!(
            se,
            "{\"Tuple\":[{\"Scalar\":\"bit\"},{\"Tuple\":[{\"Scalar\":\"bit\"}]}]}"
        );
        let de: Type = serde_json::from_str(&se).unwrap();
        assert_eq!(t, de);
    }

    #[test]
    fn test_scalar_type_fmt_display() {
        let a: ScalarType = UINT64;
        let b: ScalarType = BIT;
        let c: ScalarType = INT32;
        assert_eq!(format!("{}", a), "u64");
        assert_eq!(format!("{}", b), "bit");
        assert_eq!(format!("{}", c), "i32");
    }

    #[test]
    fn test_type_fmt_display() {
        // Testing Scalar(ScalarType),
        let a = scalar_type(UINT16);
        let b = scalar_type(BIT);
        assert_eq!(format!("{}", a), "u16");
        assert_eq!(format!("{}", b), "bit");
        // Array(ArrayShape, ScalarType),
        let array1 = array_type(vec![5, 10, 15], INT8);
        assert_eq!(format!("{}", array1), "i8[5, 10, 15]");
        let array2 = array_type(vec![], UINT32);
        assert_eq!(format!("{}", array2), "u32[]");
        // Vector(u64, TypePointer),
        let v1 = vector_type(0, a);
        assert_eq!(format!("{}", v1), "<u16{0}>");
        let v2 = vector_type(10, array1.clone());
        assert_eq!(format!("{}", v2), "<i8[5, 10, 15]{10}>");
        // Tuple(Vec<TypePointer>),
        let t1 = tuple_type(vec![]);
        assert_eq!(format!("{}", t1), "()");
        let t2 = tuple_type(vec![
            v2.clone(),
            array1,
            b,
            tuple_type(vec![vector_type(10, scalar_type(INT32)), v2]),
        ]);
        assert_eq!(
            format!("{}", t2),
            "(<i8[5, 10, 15]{10}>, i8[5, 10, 15], bit, (<i32{10}>, <i8[5, 10, 15]{10}>))"
        );
        let t3 = tuple_type(vec![
            t1,
            named_tuple_type(vec![
                ("City".to_owned(), array_type(vec![64], UINT8)),
                ("Zip".to_owned(), scalar_type(UINT32)),
            ]),
        ]);
        assert_eq!(
            format!("{}", t3),
            "((), (\\\"City\\\": u8[64], \\\"Zip\\\": u32))"
        );
        // NamedTuple(Vec<(String, TypePointer)>),
        let nt1 = named_tuple_type(vec![
            ("Name".to_owned(), array_type(vec![100], INT8)),
            ("DoB".to_owned(), array_type(vec![10], INT8)),
        ]);
        assert_eq!(
            format!("{}", nt1),
            "(\\\"Name\\\": i8[100], \\\"DoB\\\": i8[10])"
        );
        let nt2 = Type::NamedTuple(vec![]);
        assert_eq!(format!("{}", nt2), "()");
    }

    #[test]
    fn test_scalar_size_in_bits() {
        let st = ScalarType::Bit;
        assert_eq!(st.size_in_bits(), 1);
        let st = ScalarType::I8;
        assert_eq!(st.size_in_bits(), 8);
        let st = ScalarType::U16;
        assert_eq!(st.size_in_bits(), 16);
        let st = ScalarType::I32;
        assert_eq!(st.size_in_bits(), 32);
        let st = ScalarType::U64;
        assert_eq!(st.size_in_bits(), 64);
    }
}
