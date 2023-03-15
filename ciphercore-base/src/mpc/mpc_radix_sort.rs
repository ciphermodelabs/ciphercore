use serde::{Deserialize, Serialize};

use crate::custom_ops::CustomOperationBody;
use crate::data_types::Type;
use crate::errors::*;
use crate::graphs::{Context, Graph};

/// Implementation of stable RadixSort from https://eprint.iacr.org/2019/695.pdf.
/// * `key` - name of the column to be sorted.
/// * `bits_group_size` - number of bits to be processed together on the one counting step, aka `L` in section 5.3 "The optimized sorting protocol".
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub(super) struct RadixSortMPC {
    pub key: String,
    pub bits_group_size: u64,
}

/// According to the paper, the optimal value is 2 or 3. TODO: benchmark it.
const OPTIMAL_BITS_GROUP_SIZE: u64 = 2;
impl RadixSortMPC {
    pub fn new(key: String) -> Self {
        Self {
            key,
            bits_group_size: OPTIMAL_BITS_GROUP_SIZE,
        }
    }
}

/// RadixSort MPC operation for public and private data with the following arguments:
/// 1. named tuple with the column `self.key` of type `BIT` and shape `[n, b]` and other columns of arbitrary type and shape `[n, ...]`.
/// 2. PRF keys for MPC multiplication (only when data is private)
#[typetag::serde]
impl CustomOperationBody for RadixSortMPC {
    fn instantiate(&self, _context: Context, _argument_types: Vec<Type>) -> Result<Graph> {
        Err(runtime_error!("RadixSortMPC is not implemented yet"))
    }

    fn get_name(&self) -> String {
        format!("RadixSortMPC(bits_group_size={})", self.bits_group_size)
    }
}
