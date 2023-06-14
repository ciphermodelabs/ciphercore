pub mod execute_main;

pub use log::debug as log_debug;

#[doc(hidden)]
#[macro_export]
macro_rules! eprint_or_log {
    ($($x:tt)*) => {
        #[cfg(feature = "stderr-to-log")]
        $crate::log_debug!($($x)*);
        #[cfg(not(feature = "stderr-to-log"))]
        eprint!($($x)*);
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! eprintln_or_log {
    ($($x:tt)*) => {
        #[cfg(feature = "stderr-to-log")]
        $crate::log_debug!($($x)*);
        #[cfg(not(feature = "stderr-to-log"))]
        eprintln!($($x)*);
    };
}
