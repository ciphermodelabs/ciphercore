extern crate ciphercore_utils;

use ciphercore_base::custom_ops::CustomOperation;
use ciphercore_base::data_types::{ScalarType, Type};
use ciphercore_base::data_values::Value;
use ciphercore_base::errors::Result;
use ciphercore_base::graphs::Operation;
use ciphercore_base::graphs::{Graph, Node, Slice, SliceElement};
use ciphercore_base::runtime_error;
use ciphercore_base::typed_value::TypedValue;
use ciphercore_utils::errors::CiphercoreErrorBody;
use ciphercore_utils::errors::CiphercoreErrorKind;
use ciphercore_utils::errors::ErrorWithBody;

//CVecVal stores an array of values of a vector's elements
#[repr(C)]
#[derive(Clone)]
pub struct CVecVal<T> {
    pub ptr: *mut T,
    pub len: usize,
}

impl<T: Copy> CVecVal<T> {
    pub(crate) fn to_vec(&self) -> Result<Vec<T>> {
        let mut v = Vec::<T>::new();
        unsafe {
            if self.len == 0 {
                return Ok(v);
            }
            if self.ptr.is_null() {
                return Err(runtime_error!("Base pointer of vector is Null"));
            }
            for i in 0..self.len {
                let x = &*self.ptr.add(i);
                v.push(*x);
            }
        };
        Ok(v)
    }
    pub(crate) fn from_vec(mut v: Vec<T>) -> CVecVal<T> {
        v.shrink_to_fit();
        let cvec = CVecVal::<T> {
            ptr: v.as_mut_ptr(),
            len: v.len(),
        };
        std::mem::forget(v);
        cvec
    }
}

fn cvec_val_destroy_helper<T>(cvec_ptr: *mut CVecVal<T>) -> CResultVal<bool> {
    let helper = || -> Result<bool> {
        unsafe {
            let cvec = Box::from_raw(cvec_ptr);
            drop(Vec::from_raw_parts(cvec.ptr, cvec.len, cvec.len));
            drop(cvec);
        }
        Ok(true)
    };
    CResultVal::new(helper())
}

#[allow(dead_code)]
fn vector_from_unsafe_cvec_val<T: Copy>(cvec_ptr: *mut CVecVal<T>) -> Result<Vec<T>> {
    unsafe_deref(cvec_ptr)?.to_vec()
}

#[no_mangle]
pub extern "C" fn cvec_u64_destroy(cvec_ptr: *mut CVecVal<u64>) -> CResultVal<bool> {
    cvec_val_destroy_helper(cvec_ptr)
}

#[no_mangle]
pub extern "C" fn cvec_cstr_destroy(cvec_ptr: *mut CVecVal<CStr>) -> CResultVal<bool> {
    cvec_val_destroy_helper(cvec_ptr)
}

//CVec stores an array of pointers to a vector's elements
#[repr(C)]
#[derive(Clone)]
pub struct CVec<T> {
    pub ptr: *mut *mut T,
    pub len: usize,
}

impl<T: Clone> CVec<T> {
    pub(crate) fn to_vec(&self) -> Result<Vec<T>> {
        let mut v = Vec::<T>::new();
        unsafe {
            if self.len == 0 {
                return Ok(v);
            }
            if self.ptr.is_null() {
                return Err(runtime_error!("Base pointer of vector is Null"));
            }
            for i in 0..self.len {
                if self.ptr.add(i).is_null() {
                    return Err(runtime_error!("Vector pointer is Null"));
                }
                let x = &**self.ptr.add(i);
                v.push((*x).clone());
            }
        };
        Ok(v)
    }
    pub(crate) fn from_vec(v: Vec<T>) -> CVec<T> {
        let mut ptr_vec: Vec<*mut T> = v.into_iter().map(|x| unsafe_ref(x)).collect();
        ptr_vec.shrink_to_fit();
        let cvec = CVec {
            ptr: ptr_vec.as_mut_ptr(),
            len: ptr_vec.len(),
        };
        std::mem::forget(ptr_vec);
        cvec
    }
}

fn _vector_from_unsafe_cvec<T: Clone>(cvec_ptr: *mut CVec<T>) -> Result<Vec<T>> {
    unsafe_deref(cvec_ptr)?.to_vec()
}

fn cvec_destroy_helper<T>(cvec_ptr: *mut CVec<T>) -> CResultVal<bool> {
    let helper = || -> Result<bool> {
        unsafe {
            let cvec = Box::from_raw(cvec_ptr);
            Vec::from_raw_parts(cvec.ptr, cvec.len, cvec.len);
            drop(cvec);
        }
        Ok(true)
    };
    CResultVal::new(helper())
}

#[no_mangle]
pub extern "C" fn cvec_type_destroy(cvec_ptr: *mut CVec<Type>) -> CResultVal<bool> {
    cvec_destroy_helper(cvec_ptr)
}
#[no_mangle]
pub extern "C" fn cvec_node_destroy(cvec_ptr: *mut CVec<Node>) -> CResultVal<bool> {
    cvec_destroy_helper(cvec_ptr)
}
#[no_mangle]
pub extern "C" fn cvec_graph_destroy(cvec_ptr: *mut CVec<Graph>) -> CResultVal<bool> {
    cvec_destroy_helper(cvec_ptr)
}
#[no_mangle]
pub extern "C" fn cvec_cslice_element_destroy(
    cvec_ptr: *mut CVec<CSliceElement>,
) -> CResultVal<bool> {
    cvec_destroy_helper(cvec_ptr)
}

pub(crate) fn unsafe_ref<T>(x: T) -> *mut T {
    Box::into_raw(Box::new(x))
}

pub(crate) fn destroy_helper<T>(ptr: *mut T) {
    unsafe {
        Box::from_raw(ptr);
    }
}

pub(crate) fn unsafe_deref<T: Clone>(ptr: *mut T) -> Result<T> {
    unsafe {
        if ptr.is_null() {
            return Err(runtime_error!("Null pointer passed by C"));
        }
        let tmp_ptr = &*ptr;
        Ok((*tmp_ptr).clone())
    }
}

pub(crate) fn unsafe_deref_const<T: Clone>(ptr: *const T) -> Result<T> {
    unsafe {
        if ptr.is_null() {
            return Err(runtime_error!("Null pointer passed by C"));
        }
        let tmp_ptr = &*ptr;
        Ok((*tmp_ptr).clone())
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CStr {
    pub ptr: *const u8,
}
impl CStr {
    pub(crate) fn to_str_slice(self) -> Result<&'static str> {
        let cs = unsafe { std::ffi::CStr::from_ptr(self.ptr as *const i8) };
        let str_slice = cs.to_str()?;
        Ok(str_slice)
    }
    pub(crate) fn to_string(self) -> Result<String> {
        let str_slice = self.to_str_slice()?;
        Ok(str_slice.to_owned())
    }
    pub(crate) fn from_string(s: String) -> Result<CStr> {
        let cs = std::ffi::CString::new(s)?;
        let p = cs.as_ptr() as *const u8;
        unsafe_ref(cs);
        Ok(CStr { ptr: p })
    }
}

#[no_mangle]
pub extern "C" fn cstr_destroy(cstr: CStr) -> CResultVal<bool> {
    let helper = || -> Result<bool> {
        unsafe {
            Box::from_raw(cstr.ptr as *mut u8);
        }
        Ok(true)
    };
    CResultVal::new(helper())
}
#[repr(C)]
pub struct CiphercoreError {
    pub kind: CiphercoreErrorKind,
    pub msg: CStr,
}

impl CiphercoreError {
    pub(crate) fn new(body: CiphercoreErrorBody) -> CiphercoreError {
        let s = format!("{}", body);
        let cs = std::ffi::CString::new(s).unwrap();
        let p = cs.as_ptr() as *const u8;
        std::mem::forget(cs);
        CiphercoreError {
            kind: (body.kind),
            msg: (CStr { ptr: p }),
        }
    }
}

pub trait CResultTrait<T> {
    fn new(res: Result<T>) -> Self;
}

// CResult returns a raw pointer in case of success and error message and type in case of error.
#[repr(C)]
pub enum CResult<T> {
    Ok(*mut T),
    Err(CiphercoreError),
}

impl<T> CResultTrait<T> for CResult<T> {
    fn new(res: Result<T>) -> CResult<T> {
        match res {
            Ok(x) => CResult::Ok(unsafe_ref(x)),
            Err(e) => CResult::Err(CiphercoreError::new(e.get_body())),
        }
    }
}

// CResultVal returns a value in case of success and error message and type in case of error.
#[repr(C)]
pub enum CResultVal<T> {
    Ok(T),
    Err(CiphercoreError),
}

impl<T> CResultTrait<T> for CResultVal<T> {
    fn new(res: Result<T>) -> CResultVal<T> {
        match res {
            Ok(x) => CResultVal::Ok(x),
            Err(e) => CResultVal::Err(CiphercoreError::new(e.get_body())),
        }
    }
}

#[repr(C)]
#[derive(Clone)]
pub struct CTypedValue {
    json: CStr,
}
impl CTypedValue {
    pub(crate) fn to_type_value(&self) -> Result<(Type, Value)> {
        let op_str_slice = self.json.to_str_slice()?;
        let tv = serde_json::from_str::<TypedValue>(op_str_slice)?;
        Ok((tv.t, tv.value))
    }
    pub(crate) fn from_type_and_value(t: Type, value: Value) -> Result<CTypedValue> {
        let tv = TypedValue::new(t, value)?;
        let jstr = CStr::from_string(serde_json::to_string(&tv)?)?;
        Ok(CTypedValue { json: jstr })
    }
}

#[repr(C)]
pub struct CCustomOperation {
    json: CStr,
}
impl CCustomOperation {
    pub(crate) fn to_custom_op(&self) -> Result<CustomOperation> {
        let op_str_slice = self.json.to_str_slice()?;
        Ok(serde_json::from_str::<CustomOperation>(op_str_slice)?)
    }
    pub(crate) fn from_custom_op(cop: CustomOperation) -> Result<CCustomOperation> {
        Ok(CCustomOperation {
            json: CStr::from_string(serde_json::to_string(&cop)?)?,
        })
    }
}

#[derive(Clone)]
#[repr(C)]
pub struct COption_i64 {
    valid: bool,
    num: i64,
}
impl COption_i64 {
    pub(crate) fn to_option(&self) -> Option<i64> {
        if self.valid {
            Some(self.num)
        } else {
            None
        }
    }
    pub(crate) fn from_option(op: Option<i64>) -> COption_i64 {
        match op {
            Some(x) => COption_i64 {
                valid: true,
                num: x,
            },
            None => COption_i64 {
                valid: false,
                num: 0,
            },
        }
    }
}
#[derive(Clone)]
#[repr(C)]
pub struct COption_i64_triplet {
    op1: COption_i64,
    op2: COption_i64,
    op3: COption_i64,
}

#[derive(Clone)]
#[repr(C)]
pub enum CSliceElement {
    SingleIndex(i64),
    SubArray(COption_i64_triplet),
    Ellipsis,
}
impl CSliceElement {
    pub(crate) fn to_slice_element(&self) -> SliceElement {
        match self {
            Self::SingleIndex(x) => SliceElement::SingleIndex(*x),
            Self::SubArray(x) => {
                SliceElement::SubArray(x.op1.to_option(), x.op2.to_option(), x.op3.to_option())
            }
            Self::Ellipsis => SliceElement::Ellipsis,
        }
    }
    pub(crate) fn from_slice_element(se: SliceElement) -> CSliceElement {
        match se {
            SliceElement::SingleIndex(x) => Self::SingleIndex(x),
            SliceElement::SubArray(x, y, z) => Self::SubArray(COption_i64_triplet {
                op1: COption_i64::from_option(x),
                op2: COption_i64::from_option(y),
                op3: COption_i64::from_option(z),
            }),
            SliceElement::Ellipsis => Self::Ellipsis,
        }
    }
}

#[repr(C)]
#[derive(Clone)]
pub struct CSlice {
    elements: CVec<CSliceElement>,
}
impl CSlice {
    pub(crate) fn to_slice(&self) -> Result<Slice> {
        let celem_vec = self.elements.to_vec()?;
        let elem_vec = celem_vec.iter().map(|x| x.to_slice_element()).collect();
        Ok(elem_vec)
    }
    pub(crate) fn from_slice(s: Slice) -> CSlice {
        let celem_vec = s
            .iter()
            .map(|x| CSliceElement::from_slice_element((*x).clone()))
            .collect();
        let celem_cvec = CVec::from_vec(celem_vec);
        CSlice {
            elements: celem_cvec,
        }
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn c_slice_destroy(cslice_ptr: *mut CSlice) {
    unsafe {
        let cslice_ref = Box::from_raw(cslice_ptr);
        let elements = cslice_ref.elements;
        let vec_elements = Vec::from_raw_parts(elements.ptr, elements.len, elements.len);
        for elem in vec_elements {
            Box::from_raw(elem);
        }
    }
}

#[repr(C)]
#[derive(Clone)]
pub struct U64TypePtrTuple {
    iv: u64,
    type_ptr: *mut Type,
}

#[repr(C)]
pub struct CStrTypePtrTuple {
    str: CStr,
    type_ptr: *mut Type,
}
#[repr(C)]
pub enum COperation {
    Input(*mut Type),
    Add,
    Subtract,
    Multiply,
    MixedMultiply,
    Dot,
    Matmul,
    Truncate(u64),
    Sum(*mut CVecVal<u64>),
    PermuteAxes(*mut CVecVal<u64>),
    Get(*mut CVecVal<u64>),
    GetSlice(*mut CSlice),
    Reshape(*mut Type),
    NOP,
    Random(*mut Type),
    PRF(*mut U64TypePtrTuple),
    Stack(*mut CVecVal<u64>),
    Constant(*mut CTypedValue),
    A2B,
    B2A(*mut ScalarType),
    CreateTuple,
    CreateNamedTuple(*mut CVecVal<CStr>),
    CreateVector(*mut Type),
    TupleGet(u64),
    NamedTupleGet(CStr),
    VectorGet,
    Zip,
    Repeat(u64),
    Call,
    Iterate,
    ArrayToVector,
    VectorToArray,
    Custom(CCustomOperation),
}
impl COperation {
    pub(crate) fn _to_operation(&self) -> Result<Operation> {
        let op = match self {
            Self::Input(t_ptr) => Operation::Input(unsafe_deref(*t_ptr)?),
            Self::Add => Operation::Add,
            Self::Subtract => Operation::Subtract,
            Self::Multiply => Operation::Multiply,
            Self::MixedMultiply => Operation::MixedMultiply,
            Self::Dot => Operation::Dot,
            Self::Matmul => Operation::Matmul,
            Self::Truncate(x) => Operation::Truncate(*x),
            Self::Sum(cvec) => Operation::Sum(vector_from_unsafe_cvec_val(*cvec)?),
            Self::PermuteAxes(cvec) => Operation::PermuteAxes(vector_from_unsafe_cvec_val(*cvec)?),
            Self::Get(cvec) => Operation::Get(vector_from_unsafe_cvec_val(*cvec)?),
            Self::GetSlice(cslice) => Operation::GetSlice(unsafe_deref(*cslice)?.to_slice()?),
            Self::Reshape(t_ptr) => Operation::Reshape(unsafe_deref(*t_ptr)?),
            Self::NOP => Operation::NOP,
            Self::Random(t_ptr) => Operation::Random(unsafe_deref(*t_ptr)?),
            Self::PRF(tuple) => {
                let tuple_safe = unsafe_deref(*tuple)?;
                Operation::PRF(tuple_safe.iv, unsafe_deref(tuple_safe.type_ptr)?)
            }
            Self::Stack(cvec) => Operation::Stack(vector_from_unsafe_cvec_val(*cvec)?),
            Self::A2B => Operation::A2B,
            Self::B2A(st_ptr) => Operation::B2A(unsafe_deref(*st_ptr)?),
            Self::CreateTuple => Operation::CreateTuple,
            Self::CreateNamedTuple(cvec_cstr) => {
                let vec_cstr = vector_from_unsafe_cvec_val(*cvec_cstr)?;
                let vec_str = vec_cstr
                    .iter()
                    .map(|x| -> Result<String> { x.to_string() })
                    .collect::<Result<Vec<String>>>()?;
                Operation::CreateNamedTuple(vec_str)
            }
            Self::CreateVector(t_ptr) => Operation::CreateVector(unsafe_deref(*t_ptr)?),
            Self::TupleGet(x) => Operation::TupleGet(*x),
            Self::NamedTupleGet(cstr) => Operation::NamedTupleGet(cstr.to_string()?),
            Self::VectorGet => Operation::VectorGet,
            Self::Zip => Operation::Zip,
            Self::Repeat(n) => Operation::Repeat(*n),
            Self::Call => Operation::Call,
            Self::Iterate => Operation::Iterate,
            Self::ArrayToVector => Operation::ArrayToVector,
            Self::VectorToArray => Operation::VectorToArray,
            Self::Custom(c_cust_op) => Operation::Custom((*c_cust_op).to_custom_op()?),
            Self::Constant(c_val) => {
                let c_val_safe = unsafe_deref(*c_val)?;
                Operation::Constant(
                    (c_val_safe).to_type_value()?.0,
                    (c_val_safe).to_type_value()?.1,
                )
            }
        };
        Ok(op)
    }
    pub(crate) fn from_operation(op: Operation) -> Result<COperation> {
        let cop = match op {
            Operation::Input(t) => Self::Input(unsafe_ref(t)),
            Operation::Add => Self::Add,
            Operation::Subtract => Self::Subtract,
            Operation::Multiply => Self::Multiply,
            Operation::MixedMultiply => Self::MixedMultiply,
            Operation::Dot => Self::Dot,
            Operation::Matmul => Self::Matmul,
            Operation::Truncate(x) => Self::Truncate(x),
            Operation::Sum(vec) => Self::Sum(unsafe_ref(CVecVal::from_vec(vec))),
            Operation::PermuteAxes(vec) => Self::PermuteAxes(unsafe_ref(CVecVal::from_vec(vec))),
            Operation::Get(vec) => Self::Get(unsafe_ref(CVecVal::from_vec(vec))),
            Operation::GetSlice(slice) => Self::GetSlice(unsafe_ref(CSlice::from_slice(slice))),
            Operation::Reshape(t) => Self::Reshape(unsafe_ref(t)),
            Operation::NOP => Self::NOP,
            Operation::Random(t) => Self::Random(unsafe_ref(t)),
            Operation::PRF(iv, t) => Self::PRF(unsafe_ref(U64TypePtrTuple {
                iv,
                type_ptr: unsafe_ref(t),
            })),
            Operation::Stack(vec) => Self::Stack(unsafe_ref(CVecVal::from_vec(vec))),
            Operation::A2B => Self::A2B,
            Operation::B2A(st) => Self::B2A(unsafe_ref(st)),
            Operation::CreateTuple => Self::CreateTuple,
            Operation::CreateNamedTuple(vec_str) => {
                let vec_cstr = vec_str
                    .iter()
                    .map(|x| -> Result<CStr> { CStr::from_string((*x).clone()) })
                    .collect::<Result<Vec<CStr>>>()?;
                let cvec_cstr = CVecVal::from_vec(vec_cstr);
                Self::CreateNamedTuple(unsafe_ref(cvec_cstr))
            }
            Operation::CreateVector(t) => Self::CreateVector(unsafe_ref(t)),
            Operation::TupleGet(x) => Self::TupleGet(x),
            Operation::NamedTupleGet(str) => Self::NamedTupleGet(CStr::from_string(str)?),
            Operation::VectorGet => Self::VectorGet,
            Operation::Zip => Self::Zip,
            Operation::Repeat(n) => Self::Repeat(n),
            Operation::Call => Self::Call,
            Operation::Iterate => Self::Iterate,
            Operation::ArrayToVector => Self::ArrayToVector,
            Operation::VectorToArray => Self::VectorToArray,
            Operation::Custom(cust_op) => Self::Custom(CCustomOperation::from_custom_op(cust_op)?),
            Operation::Constant(t, v) => {
                Self::Constant(unsafe_ref(CTypedValue::from_type_and_value(t, v)?))
            }
        };
        Ok(cop)
    }
}

#[no_mangle]
pub extern "C" fn c_operation_destroy(cop_ptr: *mut COperation) {
    destroy_helper(cop_ptr);
}
