//! Wrapper of the Rust [Result](https://doc.rust-lang.org/std/result/) type within CipherCore used for error handling.
use ciphercore_utils::errors::{CiphercoreErrorBody, ErrorWithBody};
use json::JsonError;
use ndarray::ShapeError;
use openssl::error::ErrorStack;
use std::num::ParseIntError;

use serde::{Deserialize, Serialize};

use std::fmt;

#[doc(hidden)]
#[derive(Debug, Serialize, Deserialize)]
pub struct CiphercoreBaseError {
    body: Box<CiphercoreErrorBody>,
}

impl CiphercoreBaseError {
    pub fn new(body: CiphercoreErrorBody) -> Self {
        Self {
            body: Box::new(body),
        }
    }

    pub fn new_box(body: Box<CiphercoreErrorBody>) -> Self {
        Self { body }
    }
}

impl ErrorWithBody for CiphercoreBaseError {
    fn get_body(self) -> Box<CiphercoreErrorBody> {
        self.body
    }
}

impl fmt::Display for CiphercoreBaseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.body.fmt(f)
    }
}

#[cfg(feature = "py-binding")]
impl std::convert::From<CiphercoreBaseError> for pyo3::PyErr {
    fn from(err: CiphercoreBaseError) -> pyo3::PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
    }
}

impl std::convert::From<CiphercoreBaseError> for fmt::Error {
    fn from(_err: CiphercoreBaseError) -> fmt::Error {
        fmt::Error::default()
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! runtime_error {
    ($($x:tt)*) => {
        $crate::errors::CiphercoreBaseError::new(ciphercore_utils::runtime_error_body!($($x)*))
    };
}

impl From<ParseIntError> for CiphercoreBaseError {
    fn from(err: ParseIntError) -> CiphercoreBaseError {
        runtime_error!("ParseIntError: {}", err)
    }
}

impl From<serde_json::Error> for CiphercoreBaseError {
    fn from(err: serde_json::Error) -> CiphercoreBaseError {
        runtime_error!("serde_json::Error: {}", err)
    }
}

impl From<std::io::Error> for CiphercoreBaseError {
    fn from(err: std::io::Error) -> CiphercoreBaseError {
        runtime_error!("std::io::Error: {}", err)
    }
}

impl From<std::num::TryFromIntError> for CiphercoreBaseError {
    fn from(err: std::num::TryFromIntError) -> CiphercoreBaseError {
        runtime_error!("std::num::TryFromIntError {}", err)
    }
}

impl From<ShapeError> for CiphercoreBaseError {
    fn from(err: ShapeError) -> CiphercoreBaseError {
        runtime_error!("NDArray shape error: {}", err)
    }
}

impl From<JsonError> for CiphercoreBaseError {
    fn from(err: JsonError) -> CiphercoreBaseError {
        runtime_error!("JSON error: {}", err)
    }
}

impl From<std::ffi::NulError> for CiphercoreBaseError {
    fn from(err: std::ffi::NulError) -> CiphercoreBaseError {
        runtime_error!("Null error: {}", err)
    }
}

impl From<std::str::Utf8Error> for CiphercoreBaseError {
    fn from(err: std::str::Utf8Error) -> CiphercoreBaseError {
        runtime_error!("Utf8Error: {}", err)
    }
}

impl From<ErrorStack> for CiphercoreBaseError {
    fn from(err: ErrorStack) -> CiphercoreBaseError {
        runtime_error!("OpenSSL error: {}", err)
    }
}
/// Result type within CipherCore that is used for error handling.
///
/// This is a wrapper of the Rust [Result](https://doc.rust-lang.org/std/result/) type that is effectively an enum with the variants, `Ok(T)` and `Err(E)`, where `E` is a CipherCore error containing lots of useful information.
pub type Result<T> = std::result::Result<T, CiphercoreBaseError>;

#[cfg(test)]
mod tests {
    use crate::{errors::CiphercoreBaseError, typed_value::TypedValue};
    #[test]
    fn test_serialization_error_conversion() {
        let s = r#"{"kind":"vector","value":[{"kind":"scalar","type":"i32","value":-123456},{"kind":"scalar","type":"u32","value":123456}]}"#;
        let serde_error = serde_json::from_str::<TypedValue>(&s);
        if let Err(e) = serde_error {
            let err = CiphercoreBaseError::from(e);
            assert!(err.to_string().find("serde_json::Error: ").is_some())
        }
    }

    #[test]
    fn error_size_should_be_small() {
        // Types like `Result<u32, Err>` takes max(4, size_of(Err)) bytes, which could be
        // quite expensive. So we wrap errors into Box.
        //
        // See more: https://rust-lang.github.io/rust-clippy/master/index.html#result_large_err
        let size = std::mem::size_of::<CiphercoreBaseError>();
        assert!(size <= 16);
    }
}
