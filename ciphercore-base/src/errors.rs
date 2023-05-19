use std::sync::Arc;

#[doc(hidden)]
#[macro_export]
macro_rules! runtime_error {
    ($($x:tt)*) => {
        $crate::errors::Error::new(anyhow::anyhow!($($x)*), true)
    };
}

#[derive(Clone)]
pub struct Error {
    // Note: we use Arc to make it clonable
    inner: Arc<anyhow::Error>,
    pub can_retry: bool,
}

impl Error {
    pub fn new(err: anyhow::Error, can_retry: bool) -> Self {
        Self {
            inner: Arc::new(err),
            can_retry,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.inner, f)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.inner, f)
    }
}

#[cfg(feature = "tonic-errors")]
impl From<Error> for tonic::Status {
    fn from(err: Error) -> Self {
        // TODO: add stacktrace to details?
        tonic::Status::new(tonic::Code::Internal, err.inner.to_string())
    }
}

#[cfg(feature = "py-binding")]
impl std::convert::From<Error> for pyo3::PyErr {
    fn from(err: Error) -> pyo3::PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
    }
}

impl<E> From<E> for Error
where
    E: Into<anyhow::Error>,
{
    fn from(error: E) -> Self {
        // TODO: downcast and check some error types to set can_retry to false
        // (e.g. for authorization errors)
        Self::new(error.into(), true)
    }
}

// You can't convert Lock errors with '?' because they are not 'static, so you can `.map_err(poisoned_mutex)` instead
pub fn poisoned_mutex<E>(_: E) -> Error {
    runtime_error!("Poisoned mutex")
}
