use ciphercore_base::custom_ops::PyBindingCustomOperation;
use ciphercore_base::data_types::{
    py_binding_array_type, py_binding_named_tuple_type, py_binding_scalar_type,
    py_binding_tuple_type, py_binding_vector_type,
};
use ciphercore_base::data_types::{
    PyBindingScalarType, PyBindingType, BIT, INT16, INT32, INT64, INT8, UINT16, UINT32, UINT64,
    UINT8,
};
use ciphercore_base::data_values::PyBindingValue;
use ciphercore_base::graphs::{
    py_binding_create_context, PyBindingContext, PyBindingGraph, PyBindingJoinType, PyBindingNode,
    PyBindingShardConfig, PyBindingSliceElement,
};
use ciphercore_base::typed_value::PyBindingTypedValue;
use numpy::PyReadonlyArrayDyn;
use pyo3::prelude::{pymodule, wrap_pyfunction, PyModule, PyResult, Python};

#[pymodule]
fn ciphercore_internal(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    macro_rules! call_serialize_to_str {
        ($n:ident, $t:ident, $v:expr) => {
            #[pyo3::pyfunction]
            fn $n(x: PyReadonlyArrayDyn<$t>) -> PyResult<String> {
                Ok(rust::serialize_to_str(x, $v)?)
            }
            m.add_function(wrap_pyfunction!($n, m)?).unwrap();
        };
    }

    call_serialize_to_str!(serialize_to_str_uint64, u64, UINT64);
    call_serialize_to_str!(serialize_to_str_uint32, u32, UINT32);
    call_serialize_to_str!(serialize_to_str_uint16, u16, UINT16);
    call_serialize_to_str!(serialize_to_str_uint8, u8, UINT8);
    call_serialize_to_str!(serialize_to_str_int64, i64, INT64);
    call_serialize_to_str!(serialize_to_str_int32, i32, INT32);
    call_serialize_to_str!(serialize_to_str_int16, i16, INT16);
    call_serialize_to_str!(serialize_to_str_int8, i8, INT8);
    call_serialize_to_str!(serialize_to_str_bool, bool, BIT);

    m.add_function(wrap_pyfunction!(py_binding_create_context, m)?)?;
    m.add_function(wrap_pyfunction!(py_binding_scalar_type, m)?)?;
    m.add_function(wrap_pyfunction!(py_binding_tuple_type, m)?)?;
    m.add_function(wrap_pyfunction!(py_binding_array_type, m)?)?;
    m.add_function(wrap_pyfunction!(py_binding_vector_type, m)?)?;
    m.add_function(wrap_pyfunction!(py_binding_named_tuple_type, m)?)?;

    m.add("BIT", PyBindingScalarType { inner: BIT })?;
    m.add("UINT8", PyBindingScalarType { inner: UINT8 })?;
    m.add("INT8", PyBindingScalarType { inner: INT8 })?;
    m.add("UINT16", PyBindingScalarType { inner: UINT16 })?;
    m.add("INT16", PyBindingScalarType { inner: INT16 })?;
    m.add("UINT32", PyBindingScalarType { inner: UINT32 })?;
    m.add("INT32", PyBindingScalarType { inner: INT32 })?;
    m.add("UINT64", PyBindingScalarType { inner: UINT64 })?;
    m.add("INT64", PyBindingScalarType { inner: INT64 })?;
    m.add_class::<PyBindingScalarType>()?;
    m.add_class::<PyBindingType>()?;
    m.add_class::<PyBindingContext>()?;
    m.add_class::<PyBindingGraph>()?;
    m.add_class::<PyBindingNode>()?;
    m.add_class::<PyBindingTypedValue>()?;
    m.add_class::<PyBindingCustomOperation>()?;
    m.add_class::<PyBindingSliceElement>()?;
    m.add_class::<PyBindingValue>()?;
    m.add_class::<PyBindingJoinType>()?;
    m.add_class::<PyBindingShardConfig>()?;
    Ok(())
}

mod rust {
    use std::ops::Not;

    use ciphercore_base::data_types::ScalarType;
    use ciphercore_base::errors::Result;
    use ciphercore_base::typed_value::TypedValue;
    use ciphercore_base::typed_value_operations::TypedValueArrayOperations;
    use numpy::PyReadonlyArrayDyn;

    pub(crate) fn serialize_to_str<
        T: numpy::Element + TryInto<u128> + Not<Output = T> + TryInto<u8> + Copy,
    >(
        x: PyReadonlyArrayDyn<T>,
        st: ScalarType,
    ) -> Result<String> {
        let array = x.as_array();
        let tv = TypedValue::from_ndarray(array.to_owned(), st)?;
        Ok(serde_json::to_string(&tv)?)
    }
}
