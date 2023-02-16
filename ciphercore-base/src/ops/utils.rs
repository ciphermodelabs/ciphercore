use std::ops::Not;

use crate::data_types::{array_type, ScalarType, Type, BIT};
use crate::data_values::Value;
use crate::errors::Result;
use crate::graphs::{Context, Graph, Node, SliceElement};
use crate::typed_value::TypedValue;

/// This function tests that two given inputs containing arrays or scalars of bitstrings
/// are compatible for binary custom operations on bits that involve broadcasting,
/// e.g. comparison and binary addition.
pub(super) fn validate_arguments_in_broadcast_bit_ops(
    arguments_types: Vec<Type>,
    custom_op_name: &str,
) -> Result<()> {
    if arguments_types.len() != 2 {
        return Err(runtime_error!("Invalid number of arguments"));
    }
    let mut are_valid_inputs: bool = true;
    let mut error_message: String = format!("{custom_op_name}: ");
    match (&arguments_types[0], &arguments_types[1]) {
        (Type::Array(shape0, scalar_type0), Type::Array(shape1, scalar_type1)) => {
            if shape0[shape0.len() - 1] != shape1[shape1.len() - 1] {
                are_valid_inputs = false;
                error_message.push_str("Input arrays' last dimensions are not the same");
            } else if *scalar_type0 != BIT {
                are_valid_inputs = false;
                error_message.push_str("Input array [0]'s ScalarType is not BIT");
            } else if *scalar_type1 != BIT {
                are_valid_inputs = false;
                error_message.push_str("Input array [1]'s ScalarType is not BIT");
            }
        }
        _ => {
            are_valid_inputs = false;
            error_message.push_str("Invalid input argument type, expected Array type");
        }
    }
    if !are_valid_inputs {
        Err(runtime_error!("{}", error_message))
    } else {
        Ok(())
    }
}

/// Panics if `x` is not an array.
pub fn pull_out_bits(x: Node) -> Result<Node> {
    let shape = x.get_type()?.get_dimensions();
    if shape.len() == 1 {
        Ok(x)
    } else {
        let mut axes_permutation = vec![shape.len() as u64 - 1];
        axes_permutation.extend(0..shape.len() as u64 - 1);
        Ok(x.permute_axes(axes_permutation)?)
    }
}

pub fn pull_out_bits_for_type(t: Type) -> Result<Type> {
    if !t.is_array() {
        return Err(runtime_error!("Expected array type"));
    }
    let shape = t.get_dimensions();
    if shape.len() == 1 {
        Ok(t)
    } else {
        let mut new_shape = vec![shape[shape.len() - 1]];
        new_shape.extend(&shape[0..shape.len() - 1]);
        Ok(array_type(new_shape, t.get_scalar_type()))
    }
}

/// Panics if `x` is not an array.
pub fn put_in_bits(x: Node) -> Result<Node> {
    let shape = x.get_type()?.get_dimensions();
    if shape.len() == 1 {
        Ok(x)
    } else {
        let mut axes_permutation: Vec<u64> = (1..shape.len()).map(|x| x as u64).collect();
        axes_permutation.push(0);
        Ok(x.permute_axes(axes_permutation)?)
    }
}

pub fn zeros(g: &Graph, t: Type) -> Result<Node> {
    g.constant(t.clone(), Value::zero_of_type(t))
}

pub fn zeros_like(x: Node) -> Result<Node> {
    zeros(&x.get_graph(), x.get_type()?)
}

// Adds several zero rows to the end or beginning of the array
pub fn extend_with_zeros(g: &Graph, x: Node, num_zero_rows: u64, in_front: bool) -> Result<Node> {
    let t = x.get_type()?;
    let st = t.get_scalar_type();
    let shape = t.get_shape();
    let last_axis = shape.len() - 1;
    let mut zeros_shape = shape[0..last_axis].to_vec();
    zeros_shape.push(num_zero_rows);
    let zero_rows = zeros(g, array_type(zeros_shape, st))?;
    if in_front {
        return g.concatenate(vec![zero_rows, x], last_axis as u64);
    }
    g.concatenate(vec![x, zero_rows], last_axis as u64)
}

pub fn constant(g: &Graph, v: TypedValue) -> Result<Node> {
    g.constant(v.t, v.value)
}

pub fn constant_scalar<T: TryInto<u64> + Not<Output = T> + TryInto<u8> + Copy>(
    g: &Graph,
    value: T,
    st: ScalarType,
) -> Result<Node> {
    constant(g, TypedValue::from_scalar(value, st)?)
}

pub fn multiply_fixed_point(node1: Node, node2: Node, precision: u64) -> Result<Node> {
    node1.multiply(node2)?.truncate(1 << precision)
}

/// Converts (individual) bits to 0/1 in arithmetic form.
pub fn single_bit_to_arithmetic(node: Node, st: ScalarType) -> Result<Node> {
    let g = node.get_graph();
    let one = constant_scalar(&g, 1, st)?;
    one.mixed_multiply(node)
}

/// Similar to `numpy.expand_dims` `<https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html>`.
///
/// Insert new axis that will appear at `axis` positions in the new shape.
///
/// Only positions in a range `[0..new_shape.len())` are valid. All positions
/// in `axis` should be distinct.
pub fn expand_dims(node: Node, axis: &[usize]) -> Result<Node> {
    if axis.is_empty() {
        return Ok(node);
    }
    let old_shape = node.get_type()?.get_shape();
    let mut new_shape = vec![1; axis.len() + old_shape.len()];
    let mut new_shape_iter = 0;
    let mut axis_iter = 0;
    for &old_dim in old_shape.iter() {
        while axis_iter != axis.len() && axis[axis_iter] <= new_shape_iter {
            new_shape_iter += 1;
            axis_iter += 1;
        }
        new_shape[new_shape_iter] = old_dim;
        new_shape_iter += 1;
    }

    let scalar = node.get_type()?.get_scalar_type();
    node.get_graph()
        .reshape(node, Type::Array(new_shape, scalar))
}

// Computes cumulative OR (from the highest index to the lowest) on the n (from n-1 to 0) first elements of the array.
// In this method it's assumed that there are no bits set in the array after the n-th element.
// The result of the function is an array looking like this: 11..1100..0.
// After i steps of the loop in j-th element there is `or` of all elements data[j:j+2^i].
// To compute the step i+1 we need to shift the array by 2^i elements and `or` it to the current.
fn cumulative_or(data: Node, n: u64) -> Result<Node> {
    let (shape, sc) = match data.get_type()? {
        Type::Array(shape, sc) => (shape, sc),
        _ => return Err(runtime_error!("Expected array type")),
    };
    let g = data.get_graph();
    let pow2 = n.next_power_of_two();
    let k = pow2.trailing_zeros();
    let mut pad_shape = shape.clone();
    // pad_shape[0] = 2^k-(shape[0] - n)
    if n > shape[0] {
        pad_shape[0] = n - shape[0] + pow2;
    } else {
        let extra_bits = shape[0] - n;
        if pow2 > extra_bits {
            pad_shape[0] = pow2 - extra_bits;
        } else {
            pad_shape[0] = 0;
        }
    }
    let data = if pad_shape[0] != 0 {
        let pad = zeros(&g, array_type(pad_shape, sc))?;
        g.concatenate(vec![data, pad], 0)?
    } else {
        data
    };
    let data = data.add(constant_scalar(&g, 1, BIT)?)?;
    let mut suffix_or = data;
    for i in 0..k {
        let shift = 2_i64.pow(i);
        suffix_or = g.multiply(
            suffix_or.get_slice(vec![SliceElement::SubArray(None, Some(-shift), None)])?,
            suffix_or.get_slice(vec![SliceElement::SubArray(Some(shift), None, None)])?,
        )?;
    }
    suffix_or.add(constant_scalar(&g, 1, BIT)?)
}

// Works only on positive integers from (0; 2^denominator_cap_2k).
// For the initial approximation of 1/n, we use would like to compute x that is different form 1/n no more than twice.
// We compute the highest bit of n and than return 1 / 2^(highest_bit + 1).
// Idea from: https://codeforces.com/blog/entry/10330?#comment-157145
// Another approach to approximate the initial value is proposed here:
// https://www.ifca.ai/pub/fc10/31_47.pdf section 3.4
// However, the current approach seems to work fine for Goldschmidt's as well as Newton's method.
pub fn inverse_initial_approximation(
    context: &Context,
    t: Type,
    denominator_cap_2k: u64,
) -> Result<Graph> {
    let sc = t.get_scalar_type();
    let g = context.create_graph()?;
    let divisor = g.input(t)?;
    let divisor_bits = pull_out_bits(divisor.a2b()?)?;
    let cum_or = cumulative_or(divisor_bits, denominator_cap_2k)?;
    let highest_one_bit_binary = g.add(
        cum_or.get_slice(vec![SliceElement::SubArray(
            None,
            Some(denominator_cap_2k as i64),
            None,
        )])?,
        cum_or.get_slice(vec![SliceElement::SubArray(
            Some(1),
            Some(denominator_cap_2k as i64 + 1),
            None,
        )])?,
    )?;
    let mut result = vec![];
    for i in 0..denominator_cap_2k {
        result.push(highest_one_bit_binary.get(vec![denominator_cap_2k - i - 1])?);
    }
    for _ in denominator_cap_2k..sc.size_in_bits() {
        result.push(zeros_like(result[0].clone())?);
    }
    let approximation = g
        .create_vector(result[0].get_type()?, result)?
        .vector_to_array()?;
    let approximation = put_in_bits(approximation)?.b2a(sc)?;

    approximation.set_as_output()?;
    g.finalize()
}
// Another incarnation of `expand_dims`, following the contract https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
// (in particular, the `axis` argument can be negative).
pub fn unsqueeze(x: Node, axis: i64) -> Result<Node> {
    let (mut shape, sc) = match x.get_type()? {
        Type::Array(shape, sc) => (shape, sc),
        Type::Scalar(sc) => (vec![], sc),
        _ => {
            return Err(runtime_error!(
                "Expected array or scalar type, got {:?}",
                x.get_type()?
            ))
        }
    };
    if axis < -(shape.len() as i64) - 1 || axis > shape.len() as i64 {
        return Err(runtime_error!(
            "Expected axis in range [{}, {}], got {}",
            -(shape.len() as i64) - 1,
            shape.len() as i64,
            axis
        ));
    }
    let pos = if axis < 0 {
        shape.len() as i64 + axis + 1
    } else {
        axis
    };
    shape.insert(pos as usize, 1);
    x.reshape(array_type(shape, sc))
}

/// Similar to `Sum`, but for multiplication. Reduces over the last dimension.
/// The use-case where it makes the most sense is when the type is BIT.
pub fn reduce_mul(node: Node) -> Result<Node> {
    custom_reduce(pull_out_bits(node)?, |first, second| first.multiply(second))
}

/// Reduces an array over the first dimension, using log number of `combine` calls.
/// `combine` receives two arrays (first, second) of same shape, and needs to return one array of
/// same shape. `combine` is assumed to be associative and commutative.
pub fn custom_reduce(node: Node, combine: impl Fn(Node, Node) -> Result<Node>) -> Result<Node> {
    custom_reduce_vec(vec![node], |first, second| {
        Ok(vec![combine(first[0].clone(), second[0].clone())?])
    })?
    .into_iter()
    .next()
    .ok_or_else(|| runtime_error!("Internal error: custom_reduce_vec returned empty vec"))
}

/// Reduces a vector of arrays sharing the same first dimension, over the first dimension, using
/// log number of `combine` calls.
/// `combine` receives two vectors (first, second) of same shape, and needs to return one such
/// vector of same shape (the same vector size, and array shapes).
/// `combine` is assumed to be associative and commutative.
pub fn custom_reduce_vec(
    mut nodes: Vec<Node>,
    combine: impl Fn(Vec<Node>, Vec<Node>) -> Result<Vec<Node>>,
) -> Result<Vec<Node>> {
    if nodes.is_empty() {
        return Err(runtime_error!("Can't reduce an empty vector"));
    }
    let ns: Vec<u64> = nodes
        .iter()
        .map(|node| Ok(node.get_type()?.get_dimensions()[0]))
        .collect::<Result<_>>()?;
    let mut n = ns[0];
    if ns.iter().any(|el| *el != n) {
        return Err(runtime_error!("All nodes must share the first dimension"));
    }

    let mut result = None;
    while n > 0 {
        if n % 2 == 1 {
            let (first, rest) = nodes
                .into_iter()
                .map(|node| {
                    Ok((
                        node.get(vec![0])?,
                        if n > 1 {
                            node.get_slice(vec![SliceElement::SubArray(Some(1), None, None)])?
                        } else {
                            // There is nothing left in the array when we remove the first row.
                            // get_slice() would then give an error, but we're done anyway, so we
                            // can assign anything here.
                            node
                        },
                    ))
                })
                .collect::<Result<Vec<(Node, Node)>>>()?
                .into_iter()
                .unzip();
            result = match result {
                None => Some(first),
                Some(result) => Some(combine(result, first)?),
            };
            nodes = rest;
            n -= 1;
        } else {
            let (half1, half2) = nodes
                .into_iter()
                .map(|node| {
                    Ok((
                        node.get_slice(vec![SliceElement::SubArray(
                            Some(0),
                            Some((n / 2) as i64),
                            None,
                        )])?,
                        node.get_slice(vec![SliceElement::SubArray(
                            Some((n / 2) as i64),
                            None,
                            None,
                        )])?,
                    ))
                })
                .collect::<Result<Vec<(Node, Node)>>>()?
                .into_iter()
                .unzip();
            nodes = combine(half1, half2)?;
            n /= 2;
        }
    }
    result.ok_or_else(|| runtime_error!("Internal error: no result"))
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array};

    use super::*;
    use crate::{
        custom_ops::run_instantiation_pass,
        data_types::{scalar_type, INT64, UINT32},
        evaluators::random_evaluate,
        graphs::{create_context, util::simple_context},
        typed_value_operations::TypedValueArrayOperations,
    };

    #[test]
    fn test_inverse_initial_approximation() -> Result<()> {
        let context = create_context()?;
        let t = scalar_type(INT64);
        let denominator_cap_2k = 15;
        let g = inverse_initial_approximation(&context, t.clone(), denominator_cap_2k)?;
        g.set_as_main()?;
        context.finalize()?;
        let context = run_instantiation_pass(context)?;
        for &val in [1, 2, 4, 5, 7, 123, 12343].iter() {
            let result = random_evaluate(
                context.get_context().get_main_graph()?,
                vec![Value::from_scalar(val, t.get_scalar_type())?],
            )?
            .to_i64(t.get_scalar_type())?;
            let expected = (val as f64).recip();
            let lower_bound = (result as f64) / (1 << denominator_cap_2k) as f64;
            let upper_bound = 2.0 * lower_bound;
            assert!(lower_bound <= expected && expected <= upper_bound);
        }
        Ok(())
    }

    fn custom_reduce_vec_helper(
        arrays: Vec<TypedValue>,
        combine: impl Fn(Vec<Node>, Vec<Node>) -> Result<Vec<Node>>,
    ) -> Result<Vec<Value>> {
        let c = simple_context(|g| {
            let inputs = arrays
                .iter()
                .map(|array| g.input(array.t.clone()))
                .collect::<Result<_>>()?;
            g.create_tuple(custom_reduce_vec(inputs, combine)?)
        })?;
        let c = run_instantiation_pass(c)?.context;
        let inputs = arrays.into_iter().map(|array| array.value).collect();
        let outputs = random_evaluate(c.get_main_graph()?, inputs)?;
        outputs.to_vector()
    }

    #[test]
    fn test_custom_reduce_vec_sum_and_multiply_rows() -> Result<()> {
        let in0 =
            TypedValue::from_ndarray(array![[1, 2, 3], [4, 5, 6], [7, 8, 9]].into_dyn(), INT64)?;
        let in1 = TypedValue::from_ndarray(array![[3, 2], [4, 3], [5, 4]].into_dyn(), INT64)?;
        let result = custom_reduce_vec_helper(vec![in0, in1], |first, second| {
            Ok(vec![
                first[0].add(second[0].clone())?,
                first[1].multiply(second[1].clone())?,
            ])
        })?;
        let out0 = result[0].to_flattened_array_u64(array_type(vec![3], INT64))?;
        let out1 = result[1].to_flattened_array_u64(array_type(vec![2], INT64))?;
        assert_eq!(out0, [12, 15, 18]);
        assert_eq!(out1, [60, 24]);
        Ok(())
    }

    fn custom_reduce_helper(
        array: TypedValue,
        combine: impl Fn(Node, Node) -> Result<Node>,
    ) -> Result<Value> {
        let c = simple_context(|g| {
            let input = g.input(array.t.clone())?;
            custom_reduce(input, combine)
        })?;
        let c = run_instantiation_pass(c)?.context;
        random_evaluate(c.get_main_graph()?, vec![array.value])
    }

    #[test]
    fn test_custom_reduce_stress_sum() -> Result<()> {
        for n in 1..=32 {
            let data: Vec<u32> = (1..=n).collect();
            let input = TypedValue::from_ndarray(Array::from(data).into_dyn(), UINT32)?;
            let result = custom_reduce_helper(input, |first, second| Ok(first.add(second)?))?;
            let output = result.to_flattened_array_u64(array_type(vec![1], UINT32))?;
            assert_eq!(output, [(n * (n + 1) / 2) as u64]);
        }
        Ok(())
    }
}
