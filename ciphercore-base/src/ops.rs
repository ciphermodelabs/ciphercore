//! Implementation of several custom operations.
//! A custom operation can be thought of as a polymorphic function, i.e., where the number of inputs and their types can vary.

pub mod adder;
mod broadcast_test;
pub mod clip;
pub mod comparisons;
pub mod fixed_precision;
pub mod goldschmidt_division;
pub mod inverse_sqrt;
pub mod long_division;
pub mod min_max;
pub mod multiplexer;
pub mod newton_inversion;
pub mod pwl;
pub mod sorting;
pub mod taylor_exponent;
#[doc(hidden)]
pub mod utils;
