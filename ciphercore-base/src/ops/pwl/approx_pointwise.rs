use crate::data_types::{array_type, scalar_size_in_bits, scalar_type, vector_type, BIT};
use crate::data_values::Value;
use crate::errors::Result;
use crate::graphs::{Node, SliceElement};
use crate::ops::utils::pull_out_bits;

pub struct PWLConfig {
    pub log_buckets: u64,
    pub flatten_left: bool,
    pub flatten_right: bool,
}

/// This helper approximates any given function with a piecewise-linear approximation.
/// It is assumed that we're operating in fixed-precision arithmetic with `precision` bits after point.
/// We're approximating the function on the segment [left, right], using 2 ** `config.log_buckets` equally-distanced control points.
/// Note that the function has to be defined everywhere, even outside the segment.
/// Behavior outside of the segment is determined by `config.flatten_left` and `config.flatten_right` (they control whether it will be approximated with a constant or linearly).
///
/// Some notes on implementation. It is optimized for performance, vectorizing operations whenever possible. It uses `Rounds(A2B) + max((config.log_buckets + 1) * Rounds(MixedMultiply), 6) + 2` network rounds.
/// It is possible to remove the `config.log_buckets` at expense of more compute, but it is probably not worth it. It is also possible to slightly improve compute at expense of 7 more rounds.
pub fn create_approximation<F>(
    x: Node,
    f: F,
    left: f32,
    right: f32,
    precision: u64,
    config: PWLConfig,
) -> Result<Node>
where
    F: Fn(f32) -> f32,
{
    let st = x.get_type()?.get_scalar_type();
    if !st.get_signed() {
        return Err(runtime_error!("Only signed types are supported"));
    }
    if right <= left {
        return Err(runtime_error!(
            "Interval boundaries [left, right] should satisfy right > left"
        ));
    }
    let log_buckets = config.log_buckets;
    if log_buckets == 0 {
        return Err(runtime_error!("log_buckets should be positive"));
    }
    let bit_len = scalar_size_in_bits(st.clone());
    if log_buckets >= bit_len - 2 {
        return Err(runtime_error!("Too many approximation buckets"));
    }

    let g = x.get_graph();

    let scale = 1 << log_buckets;
    let mut xs = vec![];
    let mut ys = vec![];
    for i in -1..(scale + 2) {
        let x = left + (right - left) * (i as f32) / (scale as f32);
        let y = f(x);
        xs.push((x * ((1 << precision) as f32)) as i64);
        ys.push((y * ((1 << precision) as f32)) as i64);
    }
    // For each segment, the approximation is f_i(x) = alpha[i] * x + beta[i].
    let mut alphas = vec![];
    let mut betas = vec![];
    for i in 1..xs.len() {
        let x0 = xs[i - 1];
        let x1 = xs[i];
        let y0 = ys[i - 1];
        let y1 = ys[i];
        let c = ((y1 - y0) << precision) / (x1 - x0);
        alphas.push(c);
        betas.push(y0 - ((c * x0) >> precision));
    }
    if config.flatten_left {
        alphas[0] = 0;
        betas[0] = ys[0];
    }
    if config.flatten_right {
        let n = alphas.len() - 1;
        alphas[n] = 0;
        betas[n] = ys[ys.len() - 2];
    }

    let alphas_arr = g.constant(
        array_type(vec![alphas.len() as u64], st.clone()),
        Value::from_flattened_array(&alphas, st.clone())?,
    )?;
    let betas_arr = g.constant(
        array_type(vec![betas.len() as u64], st.clone()),
        Value::from_flattened_array(&betas, st.clone())?,
    )?;
    // We compute potential values for all segments with broadcasting.
    let mut x_shape = x.get_type()?.get_dimensions();
    x_shape.push(1);
    let expanded_x = x.reshape(array_type(x_shape, st.clone()))?;
    let all_vals = expanded_x
        .multiply(alphas_arr)?
        .truncate(1 << precision)?
        .add(betas_arr)?;
    // Bring the dimension with segments to the front.
    let mut perm: Vec<u64> = (0..all_vals.get_type()?.get_shape().len())
        .map(|x| x as u64)
        .collect();
    perm.rotate_right(1);
    let potential_vals = all_vals.permute_axes(perm)?;

    let left_fp = (left * ((1 << precision) as f32)) as i64;
    let left_node = g.constant(scalar_type(st.clone()), Value::from_scalar(left_fp, st)?)?;
    // We want to linearly transform `x` so that `left` becomes 0, and `right` becomes `scale` (as integer).
    let shifted_x = x.subtract(left_node)?;
    // We need to find `divisor` so that `(right - left) / divisor` = scale.
    // So divisor is `(right - left) / scale` (with the fixed-precision multiplier).
    let divisor = if log_buckets <= precision {
        ((right - left) * ((1 << (precision - log_buckets)) as f32)) as u64
    } else {
        ((right - left) as u64) >> (log_buckets - precision)
    };
    let scaled_x = shifted_x.truncate(divisor)?;

    let bits = pull_out_bits(scaled_x.a2b()?)?;
    // Now we're interested in the last `log_buckets` of clipped x, so we have to check msb and higher bits to see if we're outside of the interval.
    let msb = bits.get(vec![bit_len - 1])?;
    let high_bits = bits.get_slice(vec![
        SliceElement::SubArray(Some(log_buckets as i64), Some((bit_len - 1) as i64), None),
        SliceElement::Ellipsis,
    ])?;
    let low_bits = bits.get_slice(vec![
        SliceElement::SubArray(Some(0), Some(log_buckets as i64), None),
        SliceElement::Ellipsis,
    ])?;

    // NOTE: we could re-arrange potential_vals in the form potential_vals'[i] = potential_vals[reversed_bits(i)], which would make vectorization more efficient (but the code will be more arcane).
    let main_result = tree_retrieve(
        low_bits,
        potential_vals.get_slice(vec![
            SliceElement::SubArray(Some(1), Some(alphas.len() as i64 - 1), None),
            SliceElement::Ellipsis,
        ])?,
    )?;
    let left_result = potential_vals.get(vec![0])?;
    let right_result = potential_vals.get(vec![alphas.len() as u64 - 1])?;

    // Let's handle left/right cases. Left is easy - `x - r` should be negative, so we look at MSB.
    // Right is a bit more tricky. We have to check that any bit higher than `log_buckets` is set and MSB is not set.
    let is_left = msb.clone();
    let one = g.ones(scalar_type(BIT))?;
    let is_right = any_bit_set(high_bits)?.multiply(msb.add(one.clone())?)?;
    let is_main = is_left.add(one.clone())?.multiply(is_right.add(one)?)?;

    let main_result_masked = main_result.mixed_multiply(is_main)?;
    let left_result_masked = left_result.mixed_multiply(is_left)?;
    let right_result_masked = right_result.mixed_multiply(is_right)?;

    // TODO: can left/right cases be handled more efficiently?
    let result = main_result_masked
        .add(left_result_masked)?
        .add(right_result_masked)?;

    Ok(result)
}

/// Note that the following function essentially accesses an item in array `val` by an index determined by the bits in `bits`.
/// Rather than doing top-down dfs like we do in plaintext, this is doing down-up updates. We'll walk the whole tree
/// either way, but down-up is more efficient in terms of graph size and vectorization.
fn tree_retrieve(bits: Node, vals: Node) -> Result<Node> {
    let n = bits.get_type()?.get_shape()[0];
    let num_vals = vals.get_type()?.get_shape()[0];
    if num_vals != (1 << n) {
        return Err(runtime_error!(
            "Logic error: number of tree leaves should be equal to 2 ** depth"
        ));
    }
    let mut data = vals;
    for bit_index in 0..n {
        let bit = bits.get(vec![bit_index])?;
        let len = 1 << (n - bit_index);
        let even = data.get_slice(vec![
            SliceElement::SubArray(Some(0), Some(len - 1), Some(2)),
            SliceElement::Ellipsis,
        ])?;
        let odd = data.get_slice(vec![
            SliceElement::SubArray(Some(1), Some(len), Some(2)),
            SliceElement::Ellipsis,
        ])?;
        data = odd.subtract(even.clone())?.mixed_multiply(bit)?.add(even)?;
    }
    data.get(vec![0])
}

/// Efficient check whether any of the bits are set.
/// Utilizes the fact that AND is not only associative but also commutative, making the graph more vectorized.
fn any_bit_set(bits: Node) -> Result<Node> {
    let g = bits.get_graph();
    let one = g.ones(scalar_type(BIT))?;
    let mut unset = bits.add(one.clone())?;
    while unset.get_type()?.get_shape()[0] > 1 {
        let n = unset.get_type()?.get_shape()[0];
        let k = n / 2;
        let half1 = unset.get_slice(vec![
            SliceElement::SubArray(Some(0), Some(k as i64), None),
            SliceElement::Ellipsis,
        ])?;
        let half2 = unset.get_slice(vec![
            SliceElement::SubArray(Some(k as i64), Some(2 * k as i64), None),
            SliceElement::Ellipsis,
        ])?;
        let reduced = half1.multiply(half2)?;
        unset = if n % 2 == 0 {
            reduced
        } else {
            let last_element = unset.get(vec![n - 1])?;
            let elements = reduced.array_to_vector()?;
            let joined_elements = g.create_tuple(vec![elements, last_element.clone()])?;
            joined_elements
                .reshape(vector_type(k + 1, last_element.get_type()?))?
                .vector_to_array()?
        };
    }
    unset.get(vec![0])?.add(one)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::custom_ops::run_instantiation_pass;
    use crate::data_types::INT64;
    use crate::evaluators::random_evaluate;
    use crate::graphs::util::simple_context;

    fn scalar_helper(arg: f32, conf: PWLConfig) -> Result<f32> {
        let c = simple_context(|g| {
            let i = g.input(scalar_type(INT64))?;
            create_approximation(i, |x| x * x, -2.0, 2.0, 10, conf)
        })?;
        let mapped_c = run_instantiation_pass(c)?;
        let result = random_evaluate(
            mapped_c.get_context().get_main_graph()?,
            vec![Value::from_scalar(
                (arg * ((1 << 10) as f32)) as i64,
                INT64,
            )?],
        )?;
        let res = result.to_i64(INT64)?;
        Ok((res as f32) / ((1 << 10) as f32))
    }

    fn array_helper(arg: Vec<f32>, shape: Vec<u64>, conf: PWLConfig) -> Result<Vec<f32>> {
        let array_t = array_type(shape, INT64);
        let c = simple_context(|g| {
            let i = g.input(array_t.clone())?;
            create_approximation(i, |x| x * x, -2.0, 2.0, 10, conf)
        })?;
        let mapped_c = run_instantiation_pass(c)?;
        let mut arr = vec![];
        for x in arg {
            arr.push((x * ((1 << 10) as f32)) as i64);
        }
        let result = random_evaluate(
            mapped_c.get_context().get_main_graph()?,
            vec![Value::from_flattened_array(&arr, INT64)?],
        )?;
        let output = result.to_flattened_array_i64(array_t)?;
        let mut res = vec![];
        for x in output {
            res.push((x as f32) / ((1 << 10) as f32));
        }
        Ok(res)
    }

    #[test]
    fn pwl_simple_test() -> Result<()> {
        for i in 0..50 {
            let x = ((i - 25) as f32) * 0.1;
            let y = scalar_helper(
                x,
                PWLConfig {
                    log_buckets: 4,
                    flatten_left: false,
                    flatten_right: false,
                },
            )?;
            let expected = x * x;
            if i < 5 || i > 45 {
                // Edges.
                assert!((expected - y).abs() < 0.2);
            } else {
                // Inner stuff.
                assert!((expected - y).abs() < 0.02);
            }
        }
        Ok(())
    }

    #[test]
    fn pwl_flat_sides_test() -> Result<()> {
        let mut prev: Option<f32> = None;
        for i in 0..50 {
            let x = ((i - 25) as f32) * 0.1;
            let y = scalar_helper(
                x,
                PWLConfig {
                    log_buckets: 4,
                    flatten_left: true,
                    flatten_right: true,
                },
            )?;
            if i >= 3 && i <= 45 {
                assert!((y - x * x).abs() < 0.1);
                prev = None;
            } else {
                if let Some(expected) = prev {
                    assert!((y - expected).abs() < 0.02);
                }
                prev = Some(y);
            }
        }
        Ok(())
    }

    #[test]
    fn pwl_array_test() -> Result<()> {
        let mut xs = vec![];
        for i in 0..50 {
            let x = ((i - 25) as f32) * 0.1;
            xs.push(x);
        }
        let ys = array_helper(
            xs.clone(),
            vec![xs.len() as u64],
            PWLConfig {
                log_buckets: 4,
                flatten_left: false,
                flatten_right: false,
            },
        )?;
        for (x, y) in xs.iter().zip(ys.iter()) {
            let expected = *x * *x;
            if *x < -2.0 || *x > 2.0 {
                // Edges.
                assert!((expected - *y).abs() < 0.2);
            } else {
                // Inner stuff.
                assert!((expected - *y).abs() < 0.02);
            }
        }
        Ok(())
    }

    #[test]
    fn pwl_array2d_test() -> Result<()> {
        let mut xs = vec![];
        for i in 0..50 {
            let x = ((i - 25) as f32) * 0.1;
            xs.push(x);
        }
        let ys = array_helper(
            xs.clone(),
            vec![(xs.len() / 10) as u64, 10],
            PWLConfig {
                log_buckets: 4,
                flatten_left: false,
                flatten_right: false,
            },
        )?;
        for (x, y) in xs.iter().zip(ys.iter()) {
            let expected = *x * *x;
            if *x < -2.0 || *x > 2.0 {
                // Edges.
                assert!((expected - *y).abs() < 0.2);
            } else {
                // Inner stuff.
                assert!((expected - *y).abs() < 0.02);
            }
        }
        Ok(())
    }
}
