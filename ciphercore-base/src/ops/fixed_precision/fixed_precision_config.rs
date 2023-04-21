use serde::{Deserialize, Serialize};

/// Configuration for fixed precision arithmetic.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize, Copy)]
pub struct FixedPrecisionConfig {
    /// Number of bits for fixed precision.
    /// Integer `x` represents `x / 2^fractional_bits`.
    pub fractional_bits: u64,

    /// Whether to perform expensive debug-only checks for fixed precision arithmetic.
    /// E.g. for multiplication, we can check for overflow.
    pub debug: bool,
}

impl Default for FixedPrecisionConfig {
    fn default() -> Self {
        Self {
            fractional_bits: 15,
            debug: false,
        }
    }
}

impl FixedPrecisionConfig {
    pub fn denominator(&self) -> u128 {
        1u128 << self.fractional_bits
    }

    pub fn denominator_f64(&self) -> f64 {
        self.denominator() as f64
    }

    pub fn new(fractional_bits: u64) -> Self {
        Self {
            fractional_bits,
            debug: false,
        }
    }

    pub fn set_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    pub fn convert_from_float(&self, x: f64) -> i64 {
        (x * self.denominator_f64()).round() as i64
    }

    pub fn convert_to_float(&self, x: i64) -> f64 {
        x as f64 / self.denominator_f64()
    }

    pub fn convert_from_floats(&self, a: &[f64]) -> Vec<i64> {
        a.iter().map(|&x| self.convert_from_float(x)).collect()
    }

    pub fn convert_to_floats(&self, a: &[i64]) -> Vec<f64> {
        a.iter().map(|&x| self.convert_to_float(x)).collect()
    }
}
