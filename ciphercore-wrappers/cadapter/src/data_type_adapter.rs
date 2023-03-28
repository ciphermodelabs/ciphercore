extern crate ciphercore_base;
use crate::adapters_utils::{
    destroy_helper, unsafe_deref, unsafe_deref_const, CResult, CResultTrait, CResultVal, CStr,
    CVec, CVecVal,
};
use ciphercore_base::data_types::{ScalarType, Type};
use ciphercore_base::errors::Result;
use ciphercore_base::{data_types, runtime_error};

#[no_mangle]
#[used]
pub static BIT: ScalarType = data_types::BIT;
#[no_mangle]
#[used]
pub static UINT8: ScalarType = data_types::UINT8;
#[no_mangle]
#[used]
pub static UINT16: ScalarType = data_types::UINT16;
#[no_mangle]
#[used]
pub static UINT32: ScalarType = data_types::UINT32;
#[no_mangle]
#[used]
pub static UINT64: ScalarType = data_types::UINT64;
#[no_mangle]
#[used]
pub static INT8: ScalarType = data_types::INT8;
#[no_mangle]
#[used]
pub static INT16: ScalarType = data_types::INT16;
#[no_mangle]
#[used]
pub static INT32: ScalarType = data_types::INT32;
#[no_mangle]
#[used]
pub static INT64: ScalarType = data_types::INT64;

fn type_method_helper<T, F, R: CResultTrait<T>>(type_ptr: *mut Type, op: F) -> R
where
    F: FnOnce(Type) -> Result<T>,
{
    let helper = || -> Result<T> {
        let t = unsafe_deref(type_ptr)?;
        op(t)
    };
    R::new(helper())
}

fn scalar_type_method_helper<T, F, R: CResultTrait<T>>(st_ptr: *const ScalarType, op: F) -> R
where
    F: FnOnce(ScalarType) -> Result<T>,
{
    let helper = || -> Result<T> {
        let st = unsafe_deref_const(st_ptr)?;
        op(st)
    };
    R::new(helper())
}

#[no_mangle]
pub extern "C" fn scalar_type_get_signed(st_ptr: *const ScalarType) -> CResultVal<bool> {
    scalar_type_method_helper(st_ptr, |st| Ok(st.is_signed()))
}

#[no_mangle]
pub extern "C" fn scalar_type_get_modulus(st_ptr: *const ScalarType) -> CResultVal<u64> {
    scalar_type_method_helper(st_ptr, |st| {
        st.get_modulus()
            .map(|x| x as u64)
            .ok_or_else(|| runtime_error!("no modulus"))
    })
}

#[no_mangle]
pub extern "C" fn scalar_type_to_string(st_ptr: *const ScalarType) -> CResultVal<CStr> {
    scalar_type_method_helper(st_ptr, |st| CStr::from_string(format!("{}", st)))
}

#[no_mangle]
pub extern "C" fn scalar_type_destroy(st_ptr: *mut ScalarType) {
    destroy_helper(st_ptr);
}
#[no_mangle]
pub extern "C" fn type_destroy(t_ptr: *mut Type) {
    destroy_helper(t_ptr);
}

#[no_mangle]
pub extern "C" fn type_is_scalar(type_ptr: *mut Type) -> CResultVal<bool> {
    type_method_helper(type_ptr, |t| Ok(t.is_scalar()))
}

#[no_mangle]
pub extern "C" fn type_is_array(type_ptr: *mut Type) -> CResultVal<bool> {
    type_method_helper(type_ptr, |t| Ok(t.is_array()))
}

#[no_mangle]
pub extern "C" fn type_is_vector(type_ptr: *mut Type) -> CResultVal<bool> {
    type_method_helper(type_ptr, |t| Ok(t.is_vector()))
}

#[no_mangle]
pub extern "C" fn type_is_tuple(type_ptr: *mut Type) -> CResultVal<bool> {
    type_method_helper(type_ptr, |t| Ok(t.is_tuple()))
}

#[no_mangle]
pub extern "C" fn type_is_named_tuple(type_ptr: *mut Type) -> CResultVal<bool> {
    type_method_helper(type_ptr, |t| Ok(t.is_named_tuple()))
}

#[no_mangle]
pub extern "C" fn type_get_scalar_type(type_ptr: *mut Type) -> CResult<ScalarType> {
    type_method_helper(type_ptr, |t| Ok(t.get_scalar_type()))
}

#[no_mangle]
pub extern "C" fn type_to_string(type_ptr: *mut Type) -> CResultVal<CStr> {
    type_method_helper(type_ptr, |t| CStr::from_string(format!("{}", t)))
}

#[no_mangle]
pub extern "C" fn type_get_shape(type_ptr: *mut Type) -> CResult<CVecVal<u64>> {
    type_method_helper(type_ptr, |t| Ok(CVecVal::from_vec(t.get_shape())))
}

#[no_mangle]
pub extern "C" fn type_get_dimensions(type_ptr: *mut Type) -> CResult<CVecVal<u64>> {
    type_method_helper(type_ptr, |t| Ok(CVecVal::from_vec(t.get_shape())))
}

#[no_mangle]
pub extern "C" fn scalar_type(st_ptr: *const ScalarType) -> CResult<Type> {
    scalar_type_method_helper(st_ptr, |st| Ok(data_types::scalar_type(st)))
}

#[no_mangle]
pub extern "C" fn array_type(shape: CVecVal<u64>, st_ptr: *const ScalarType) -> CResult<Type> {
    scalar_type_method_helper(st_ptr, |st| Ok(data_types::array_type(shape.to_vec()?, st)))
}

#[no_mangle]
pub extern "C" fn vector_type(n: u64, t_ptr: *mut Type) -> CResult<Type> {
    type_method_helper(t_ptr, |t| Ok(data_types::vector_type(n, t)))
}

#[no_mangle]
pub extern "C" fn tuple_type(type_ptrs_cvec: CVec<Type>) -> CResult<Type> {
    let helper = || -> Result<Type> {
        let vt = type_ptrs_cvec.to_vec()?;
        Ok(data_types::tuple_type(vt))
    };
    CResult::new(helper())
}

#[no_mangle]
pub extern "C" fn named_tuple_type(
    cstr_cvec: CVecVal<CStr>,
    type_ptrs_cvec: CVec<Type>,
) -> CResult<Type> {
    let helper = || -> Result<Type> {
        let vt = type_ptrs_cvec.to_vec()?;
        let cstr_vec = cstr_cvec.to_vec()?;
        let str_vec: Vec<String> = cstr_vec
            .iter()
            .map(|x| -> Result<String> { x.to_string() })
            .collect::<Result<Vec<String>>>()?;
        let elem: Vec<(String, Type)> = str_vec
            .iter()
            .zip(vt.iter())
            .map(|(x, y)| ((*x).clone(), (*y).clone()))
            .collect();
        Ok(data_types::named_tuple_type(elem))
    };
    CResult::new(helper())
}

#[no_mangle]
pub extern "C" fn scalar_size_in_bits(st_ptr: *const ScalarType) -> CResultVal<u64> {
    scalar_type_method_helper(st_ptr, |st| Ok(data_types::scalar_size_in_bits(st)))
}
#[no_mangle]
pub extern "C" fn get_size_in_bits(t_ptr: *mut Type) -> CResultVal<u64> {
    type_method_helper(t_ptr, data_types::get_size_in_bits)
}

#[no_mangle]
pub extern "C" fn get_types_vector(t_ptr: *mut Type) -> CResult<CVec<Type>> {
    type_method_helper(t_ptr, |t| {
        let type_ptr_vec = data_types::get_types_vector(t)?;
        let type_vec = type_ptr_vec.iter().map(|x| (**x).clone()).collect();
        Ok(CVec::from_vec(type_vec))
    })
}
