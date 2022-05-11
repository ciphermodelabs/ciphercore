//! TODO: high-level library documentation
#![cfg_attr(feature = "nightly-features", feature(backtrace))]

#[macro_use]
pub mod errors;
pub mod applications;
#[doc(hidden)]
pub mod broadcast;
#[doc(hidden)]
pub mod bytes;
mod constants;
pub mod custom_ops;
pub mod data_types;
pub mod data_values;
#[doc(hidden)]
pub mod evaluators;
pub mod graphs;
#[doc(hidden)]
pub mod inline;
#[doc(hidden)]
pub mod mpc;
pub mod ops;
#[doc(hidden)]
pub mod optimizer;
#[doc(hidden)]
pub mod random;
#[doc(hidden)]
pub mod slices;
#[doc(hidden)]
pub mod type_inference;
#[doc(hidden)]
pub mod typed_value;
#[doc(hidden)]
pub mod version;

#[cfg(test)]
#[macro_use]
extern crate maplit;
