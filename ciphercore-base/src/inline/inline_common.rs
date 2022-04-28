use crate::errors::Result;
use crate::graphs::{Graph, Node};
use crate::inline::data_structures::{
    prefix_sums_binary_ascent, prefix_sums_segment_tree, prefix_sums_sqrt_trick, CombineOp,
};
use serde::{Deserialize, Serialize};

// This trait is needed only for calling back to the inlining processor from the
// individual inliners, and mocking it out in the tests/
pub(super) trait InlineState {
    fn assign_input_nodes(&mut self, graph: Graph, nodes: Vec<Node>) -> Result<()>;
    fn unassign_nodes(&mut self, graph: Graph) -> Result<()>;
    fn recursively_inline_graph(&mut self, graph: Graph) -> Result<Node>;
    fn output_graph(&self) -> Graph;
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum DepthOptimizationLevel {
    Default,
    // The "Extreme" level will aggressively trade performance for lower depth.
    Extreme,
}

pub(super) type PrefixSumAlgorithm<T> = fn(&[T], &mut dyn CombineOp<T>) -> Result<Vec<T>>;

pub(super) fn pick_prefix_sum_algorithm<T: std::clone::Clone>(
    inputs_len: u64,
    optimization_level: DepthOptimizationLevel,
) -> PrefixSumAlgorithm<T> {
    if matches!(optimization_level, DepthOptimizationLevel::Extreme) {
        // Get best depth possible.
        prefix_sums_binary_ascent
    } else {
        // Performance matters, use O(n)-complexity algorithms.
        // Why 16? This is the point where the segment tree inlining (2 * log(n) depth)
        // becomes better than sqrt-inlining (2 * sqrt(n) depth).
        if inputs_len < 16 {
            prefix_sums_sqrt_trick
        } else {
            prefix_sums_segment_tree
        }
    }
}
