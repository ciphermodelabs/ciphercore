use crate::data_types::{ScalarType, Type};
use crate::errors::Result;
use std::ops::Not;
use std::option::Option;

/// Converts `self` to a multi-dimensional array with the corresponding type.
///
/// # Result
///
/// Resulting multi-dimensional array with the corresponding type.
///
/// # Examples
///
/// ```
/// # use ciphercore_base::data_types::{INT32, array_type};
/// # use ciphercore_base::typed_value::TypedValue;
/// # use ndarray::array;
/// # use ciphercore_base::typed_value_operations::{TypedValueArrayOperations,ToNdarray};
/// let a = array![[-123, 123], [-456, 456]].into_dyn();
/// let v = TypedValue::from_ndarray(a, INT32).unwrap();
/// let a = ToNdarray::<i32>::to_ndarray(&v).unwrap();
/// assert_eq!(a, array![[-123i32, 123i32], [-456i32, 456i32]].into_dyn());
/// ```
pub trait ToNdarray<T> {
    fn to_ndarray(&self) -> Result<ndarray::ArrayD<T>>;
}

pub trait TypedValueArrayOperations<T>:
    ToNdarray<bool>
    + ToNdarray<u8>
    + ToNdarray<u16>
    + ToNdarray<u32>
    + ToNdarray<u64>
    + ToNdarray<i8>
    + ToNdarray<i16>
    + ToNdarray<i32>
    + ToNdarray<i64>
{
    fn from_ndarray<ST: TryInto<u64> + Not<Output = ST> + TryInto<u8> + Copy>(
        a: ndarray::ArrayD<ST>,
        st: ScalarType,
    ) -> Result<T>;
}
pub enum FromVectorMode {
    Vector,
    Tuple,
    AutoDetetion,
}
pub trait TypedValueOperations<T>: TypedValueArrayOperations<T> {
    fn get_type(&self) -> Type;
    fn is_equal(&self, other: &T) -> Result<bool>;
    fn to_vector(&self) -> Result<Vec<T>>;
    fn from_vector(v: Vec<T>, mode: FromVectorMode) -> Result<T>;
    fn get(&self, index: usize) -> Result<T>;
    fn get_sub_vector(
        &self,
        start_index_option: Option<usize>,
        end_index_option: Option<usize>,
        step_option: Option<usize>,
    ) -> Result<T>;
    // Please note that call to this function copies all the data, as a result pushing N elements
    // has quadratic complexity.
    fn push(&mut self, to_push_element: T) -> Result<()>;
    fn insert(&mut self, to_insert_element: T, index: usize) -> Result<()>;
    fn extend(&mut self, to_extend_collection: T) -> Result<()>;
    fn remove(&mut self, index: usize) -> Result<()>;
    fn pop(&mut self, index: usize) -> Result<T>;
}
pub(crate) fn get_helper<T>(tv: &T, index: usize) -> Result<T>
where
    T: TypedValueOperations<T> + Clone,
{
    let v = tv.to_vector()?;
    if index >= v.len() {
        return Err(runtime_error!("Out of bound get"));
    }
    Ok(v[index].clone())
}

fn from_vector_helper<T>(tv: &T, v: Vec<T>) -> Result<T>
where
    T: TypedValueOperations<T>,
{
    match tv.get_type() {
        Type::Tuple(_) | Type::NamedTuple(_) => T::from_vector(v, FromVectorMode::Tuple),
        Type::Vector(_, _) => T::from_vector(v, FromVectorMode::Vector),
        _ => Err(runtime_error!("Not a vector!")),
    }
}

pub(crate) fn get_sub_vector_helper<T>(
    tv: &T,
    start_index_option: Option<usize>,
    end_index_option: Option<usize>,
    step_option: Option<usize>,
) -> Result<T>
where
    T: TypedValueOperations<T> + Clone,
{
    let mut v = tv.to_vector()?;
    let start_index = start_index_option.unwrap_or(0);
    let end_index = end_index_option.unwrap_or(v.len() - 1);
    let step = step_option.unwrap_or(1);
    if start_index >= v.len() || end_index >= v.len() {
        return Err(runtime_error!("Out of bound get"));
    }
    v.truncate(end_index);
    let new_v: Vec<T> = v.iter().skip(start_index).step_by(step).cloned().collect();
    from_vector_helper(tv, new_v)
}
pub(crate) fn insert_helper<T>(tv: &T, to_insert_element: T, index: usize) -> Result<T>
where
    T: TypedValueOperations<T>,
{
    let mut v = tv.to_vector()?;
    if index > v.len() {
        return Err(runtime_error!("Out of bound insert"));
    }
    v.insert(index, to_insert_element);
    from_vector_helper(tv, v)
}
pub(crate) fn push_helper<T>(tv: &T, to_insert_element: T) -> Result<T>
where
    T: TypedValueOperations<T>,
{
    insert_helper::<T>(tv, to_insert_element, tv.to_vector()?.len())
}

pub(crate) fn extend_helper<T>(tv: &T, to_extend_collection: T) -> Result<T>
where
    T: TypedValueOperations<T>,
{
    let mut v = tv.to_vector()?;
    let v_to_extend = to_extend_collection.to_vector()?;
    v.extend(v_to_extend);
    from_vector_helper(tv, v)
}

pub(crate) fn remove_helper<T>(tv: &T, index: usize) -> Result<T>
where
    T: TypedValueOperations<T>,
{
    let mut v = tv.to_vector()?;
    if index >= v.len() {
        return Err(runtime_error!("Out of bound insert"));
    }
    v.remove(index);
    from_vector_helper(tv, v)
}

pub(crate) fn pop_helper<T>(tv: &T, index: usize) -> Result<(T, T)>
where
    T: TypedValueOperations<T> + Clone,
{
    let ret1 = get_helper::<T>(tv, index)?;
    let ret2 = remove_helper::<T>(tv, index)?;
    Ok((ret1, ret2))
}
