//! Wrapper for a main function to run CipherCore
use crate::errors::ErrorWithBody;
use log::error;
#[cfg(feature = "nightly-features")]
use log::info;
use std::fmt::Display;
use std::process;
use std::result::Result;

/// Executes CipherCore code such that all the internal errors are properly formatted and logged.
///
/// # Arguments
///
/// `f` - closure without arguments containing CipherCore code and returning `Result<()>`
pub fn execute_main<T, E>(f: T)
where
    T: FnOnce() -> Result<(), E> + std::panic::UnwindSafe,
    E: ErrorWithBody + Display,
{
    let result = std::panic::catch_unwind(|| {
        let result = f();
        if let Err(e) = result {
            error!("CipherCore Error: {}", e);
            #[cfg(feature = "nightly-features")]
            info!("error backtrace: \n{}", e.get_body().backtrace);
            process::exit(1);
        }
    });
    process_result(result);
}

#[doc(hidden)]
pub fn extract_panic_message(e: Box<dyn std::any::Any + Send>) -> Option<String> {
    match e.downcast::<String>() {
        Ok(panic_msg) => Some(*panic_msg),
        Err(e) => match e.downcast::<&str>() {
            Ok(panic_msg) => Some((*panic_msg).to_owned()),
            Err(_) => None,
        },
    }
}

#[doc(hidden)]
pub fn process_result<R>(result: std::thread::Result<R>) {
    if let Err(e) = result {
        match extract_panic_message(e) {
            Some(panic_msg) => error!("panic: {}", panic_msg),
            None => error!("panic of unknown type"),
        }
        process::exit(1);
    }
}
