use serde::{Deserialize, Serialize};

/// Configuration for fixed precision arithmetic.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize, Copy)]
pub struct FixedPrecisionConfig {
    /// Number of binary points for fixed precision.
    /// Integer `x` represents `x / 2^points`.
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
    pub fn denominator(&self) -> u64 {
        1 << self.fractional_bits
    }
}
