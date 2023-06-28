use crate::errors::Result;

pub(super) trait CombineOp<T> {
    fn combine(&mut self, arg1: T, arg2: T) -> Result<T>;
}

// Simple sum computation in the form of binary tree. `combine_op` must be associative.
// Depth: RoundUp(log(len))
// Complexity: len - 1
pub(super) fn log_depth_sum<T: std::clone::Clone>(
    items: &[T],
    combine_op: &mut dyn CombineOp<T>,
) -> Result<T> {
    if items.is_empty() {
        return Err(runtime_error!("Cannot combine empty vector"));
    }
    let mut combined_items = items.to_owned();
    while combined_items.len() > 1 {
        let mut new_combined_items = vec![];
        for i in (0..combined_items.len()).step_by(2) {
            let j = i + 1;
            if j >= combined_items.len() {
                new_combined_items.push(combined_items[i].clone());
            } else {
                let new_item =
                    combine_op.combine(combined_items[i].clone(), combined_items[j].clone())?;
                new_combined_items.push(new_item);
            }
        }
        combined_items = new_combined_items;
    }
    Ok(combined_items[0].clone())
}

/// Computes prefix sums for the given vector. `combine_op` must be associative.
/// Returned vector contains sums of `items[0]..items[i]` at position i.
/// Depth: RoundUp(log(len))
/// Complexity: len * RoundUp(log(len))
#[allow(dead_code)]
pub(super) fn prefix_sums_binary_ascent<T: std::clone::Clone>(
    items: &[T],
    combine_op: &mut dyn CombineOp<T>,
) -> Result<Vec<T>> {
    if items.is_empty() {
        return Ok(vec![]);
    }
    let mut combined_items = items.to_owned();
    let mut depth = 1;
    // Invariant: combined_items[i] = sum(items[max(i - depth + 1, 0) : i + 1])
    while depth < combined_items.len() {
        for i in (depth..combined_items.len()).rev() {
            combined_items[i] =
                combine_op.combine(combined_items[i - depth].clone(), combined_items[i].clone())?;
        }
        depth *= 2;
    }
    Ok(combined_items)
}

/// Computes prefix sums for the given vector. `combine_op` must be associative.
/// Returned vector contains sums of `items[0]..items[i]` at position i.
/// Depth: 2 * RoundUp(sqrt(len))
/// Complexity: 2 * len
#[allow(dead_code)]
pub(super) fn prefix_sums_sqrt_trick<T: std::clone::Clone>(
    items: &[T],
    combine_op: &mut dyn CombineOp<T>,
) -> Result<Vec<T>> {
    if items.is_empty() {
        return Ok(vec![]);
    }
    let block_size = std::cmp::max(1, (items.len() as f64).sqrt() as usize);
    let mut combined_items = items.to_owned();
    // Invariant: combined_items[i] = sum(items[i - i % block_size : i + 1])
    for i in 0..combined_items.len() {
        if i % block_size != 0 {
            combined_items[i] =
                combine_op.combine(combined_items[i - 1].clone(), combined_items[i].clone())?;
        }
    }
    // Now, compute the actual sums.
    for i in block_size..combined_items.len() {
        combined_items[i] = combine_op.combine(
            combined_items[i - i % block_size - 1].clone(),
            combined_items[i].clone(),
        )?;
    }
    Ok(combined_items)
}

/// Computes prefix sums for the given vector. `combine_op` must be associative.
/// Returned vector contains sums of `items[0]..items[i]` at position i.
/// Depth: RoundUp(log(len)) * 2
/// Complexity: len * 2
#[allow(dead_code)]
pub(super) fn prefix_sums_segment_tree<T: std::clone::Clone>(
    items: &[T],
    combine_op: &mut dyn CombineOp<T>,
) -> Result<Vec<T>> {
    if items.is_empty() {
        return Ok(vec![]);
    }
    // We construct segment tree layer by layer.
    let mut layers = vec![items.to_owned()];
    let mut layer = 0;
    while layers[layer].len() > 1 {
        let mut next_layer = vec![];
        for i in (0..layers[layer].len()).step_by(2) {
            if i + 1 < layers[layer].len() {
                next_layer.push(
                    combine_op.combine(layers[layer][i].clone(), layers[layer][i + 1].clone())?,
                );
            } else {
                next_layer.push(layers[layer][i].clone());
            }
        }
        layer += 1;
        layers.push(next_layer);
    }
    // Now, we go from top to bottom and compute prefix sums in each layer.
    for i in (0..layers.len() - 1).rev() {
        for j in 1..layers[i].len() {
            if j % 2 == 1 {
                layers[i][j] = layers[i + 1][j / 2].clone();
            } else {
                layers[i][j] =
                    combine_op.combine(layers[i + 1][(j - 1) / 2].clone(), layers[i][j].clone())?;
            }
        }
    }
    Ok(layers[0].clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct IntCombiner {}

    impl CombineOp<u64> for IntCombiner {
        fn combine(&mut self, arg1: u64, arg2: u64) -> Result<u64> {
            Ok(arg1 + arg2)
        }
    }

    #[test]
    fn test_log_depth_sum() {
        for len in 1..20 {
            let mut v = vec![];
            for i in 1..len + 1 {
                v.push(i * i);
            }
            let mut expected = 0;
            for x in v.clone() {
                expected += x;
            }
            let mut combiner = IntCombiner {};
            let actual = log_depth_sum(&v, &mut combiner).unwrap();
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn test_prefix_sums() {
        for len in 0..20 {
            let mut v = vec![];
            for i in 1..len + 1 {
                v.push(i * i);
            }
            let mut expected = vec![];
            let mut sum = 0;
            for x in v.clone() {
                sum += x;
                expected.push(sum);
            }
            let mut combiner = IntCombiner {};
            let actual_bin_ascent = prefix_sums_binary_ascent(&v, &mut combiner).unwrap();
            assert_eq!(expected, actual_bin_ascent);
            let actual_sqrt = prefix_sums_sqrt_trick(&v, &mut combiner).unwrap();
            assert_eq!(expected, actual_sqrt);
            let actual_fenwick = prefix_sums_segment_tree(&v, &mut combiner).unwrap();
            assert_eq!(expected, actual_fenwick);
        }
    }

    // Rather than doing anything useful, this combiner tracks number of operations and total depth.
    struct TrackingCombiner {
        total_calls: u64,
    }

    impl CombineOp<u64> for TrackingCombiner {
        fn combine(&mut self, arg1: u64, arg2: u64) -> Result<u64> {
            self.total_calls += 1;
            Ok(std::cmp::max(arg1, arg2) + 1) // Return new depth.
        }
    }

    #[test]
    fn test_complexity() {
        for len in 200..300 {
            let v = vec![0; len];
            {
                let mut combiner = TrackingCombiner { total_calls: 0 };
                let depth = log_depth_sum(&v, &mut combiner).unwrap();
                assert!(depth <= (len as f64).log(2.0).ceil() as u64);
                assert!(combiner.total_calls < len as u64);
            }
            {
                let mut combiner = TrackingCombiner { total_calls: 0 };
                let depths = prefix_sums_binary_ascent(&v, &mut combiner).unwrap();
                let log_n: u64 = (len as f64).log(2.0).ceil() as u64;
                assert!(depths.iter().max().unwrap() <= &log_n);
                assert!(combiner.total_calls <= (len as u64) * log_n);
            }
            {
                let mut combiner = TrackingCombiner { total_calls: 0 };
                let depths = prefix_sums_sqrt_trick(&v, &mut combiner).unwrap();
                let sqrt_n: u64 = (len as f64).sqrt().ceil() as u64;
                assert!(depths.iter().max().unwrap() <= &(2 * sqrt_n + 1));
                assert!(combiner.total_calls <= (len as u64) * 2);
            }
            {
                let mut combiner = TrackingCombiner { total_calls: 0 };
                let depths = prefix_sums_segment_tree(&v, &mut combiner).unwrap();
                let log_n: u64 = (len as f64).log(2.0).ceil() as u64;
                assert!(depths.iter().max().unwrap() <= &(2 * log_n + 1));
                assert!(combiner.total_calls <= (len as u64) * 2);
            }
        }
    }
}
