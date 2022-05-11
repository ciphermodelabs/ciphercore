//! Wrapper of the Rust [Result](https://doc.rust-lang.org/std/result/) type within CipherCore used for error handling.
use ciphercore_utils::errors::{CiphercoreErrorBody, ErrorWithBody};
use json::JsonError;
use ndarray::ShapeError;
use std::num::ParseIntError;

use serde::{Deserialize, Serialize};

use std::fmt;

#[doc(hidden)]
#[derive(Debug, Serialize, Deserialize)]
pub struct CiphercoreBaseError {
    body: CiphercoreErrorBody,
}

impl CiphercoreBaseError {
    pub fn new(body: CiphercoreErrorBody) -> Self {
        Self { body }
    }
}

impl ErrorWithBody for CiphercoreBaseError {
    fn get_body(&self) -> CiphercoreErrorBody {
        self.body.clone()
    }
}

impl fmt::Display for CiphercoreBaseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", serde_json::to_string_pretty(&self).unwrap())
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
        let err_str = err.to_string();
        serde_json::from_str::<CiphercoreBaseError>(&err_str)
            .expect("Error during error conversion form serde_json error to CiphercoreBaseError")
    }
}

impl From<std::io::Error> for CiphercoreBaseError {
    fn from(err: std::io::Error) -> CiphercoreBaseError {
        runtime_error!("std::io::Error: {}", err)
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
/// Result type within CipherCore that is used for error handling.
///
/// This is a wrapper of the Rust [Result](https://doc.rust-lang.org/std/result/) type that is effectively an enum with the variants, `Ok(T)` and `Err(E)`, where `E` is a CipherCore error containing lots of useful information.
pub type Result<T> = std::result::Result<T, CiphercoreBaseError>;

#[cfg(test)]
mod tests {
    use crate::errors::CiphercoreBaseError;
    use serde_json::Value;
    #[test]
    fn test_serialization_error_conversion() {
        let orignal_error = runtime_error!("Value version doesn't match the requirement");
        let orignal_error_str = orignal_error.to_string();
        let serde_error: Result<Value, serde_json::Error> =
            Err(orignal_error).map_err(serde::de::Error::custom);
        if let Err(e) = serde_error {
            let cipher_err = CiphercoreBaseError::from(e);
            assert_eq!(orignal_error_str, cipher_err.to_string());
        }
    }
}
