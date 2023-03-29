#[cfg(feature = "fuzzing")]
pub mod type_size_limit_constants {
    pub const MAX_TOTAL_SIZE_NODES: u64 = 10000;
    pub const MAX_INDIVIDUAL_NODE_SIZE: u64 = 1000;
    pub const TYPES_VECTOR_LENGTH_LIMIT: usize = 1000;
    pub const TYPE_MEMORY_OVERHEAD: u64 = 1;
}
#[cfg(not(feature = "fuzzing"))]
pub mod type_size_limit_constants {
    pub const MAX_TOTAL_SIZE_NODES: u64 = u64::MAX - 1;
    pub const MAX_INDIVIDUAL_NODE_SIZE: u64 = u64::MAX - 1;
    pub const TYPES_VECTOR_LENGTH_LIMIT: usize = usize::MAX - 1;
    pub const TYPE_MEMORY_OVERHEAD: u64 = 1;
}
