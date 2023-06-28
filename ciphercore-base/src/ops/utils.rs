use std::ops::Not;

use crate::custom_ops::CustomOperation;
use crate::data_types::{array_type, scalar_type, ArrayShape, ScalarType, Type, BIT};
use crate::errors::Result;
use crate::graphs::{Context, Graph, Node, SliceElement};
use crate::typed_value::TypedValue;

use super::goldschmidt_division::GoldschmidtDivision;

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

// Like pull_out_bits but for a pair of nodes, aligning the "bits dimension" before pulling out.
pub fn pull_out_bits_pair(x: Node, y: Node) -> Result<(Node, Node)> {
    let num_dims = std::cmp::max(
        x.get_type()?.get_shape().len(),
        y.get_type()?.get_shape().len(),
    );
    Ok((
        pull_out_bits(reshape_prepending_dims(x, num_dims)?)?,
        pull_out_bits(reshape_prepending_dims(y, num_dims)?)?,
    ))
}

pub fn prepend_dims(shape: ArrayShape, num_dims: usize) -> Result<ArrayShape> {
    match shape.len() {
        len if len == num_dims => Ok(shape),
        len if len < num_dims => Ok([vec![1; num_dims - shape.len()], shape].concat()),
        _ => Err(runtime_error!(
            "prepend_dims(num_dims={num_dims}): input shape {shape:?} too large"
        )),
    }
}

pub fn reshape_prepending_dims(node: Node, num_dims: usize) -> Result<Node> {
    let t = node.get_type()?;
    let shape = t.get_shape();
    let new_shape = prepend_dims(shape.clone(), num_dims)?;
    if shape == new_shape {
        Ok(node)
    } else {
        Ok(node.reshape(array_type(new_shape, t.get_scalar_type()))?)
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

pub fn zeros_like(x: Node) -> Result<Node> {
    x.get_graph().zeros(x.get_type()?)
}

pub fn ones_like(x: Node) -> Result<Node> {
    x.get_graph().ones(x.get_type()?)
}

// Adds several zero rows to the end or beginning of the array
pub fn extend_with_zeros(g: &Graph, x: Node, num_zero_rows: u64, in_front: bool) -> Result<Node> {
    let t = x.get_type()?;
    let st = t.get_scalar_type();
    let shape = t.get_shape();
    let last_axis = shape.len() - 1;
    let mut zeros_shape = shape[0..last_axis].to_vec();
    zeros_shape.push(num_zero_rows);
    let zero_rows = g.zeros(array_type(zeros_shape, st))?;
    if in_front {
        return g.concatenate(vec![zero_rows, x], last_axis as u64);
    }
    g.concatenate(vec![x, zero_rows], last_axis as u64)
}

pub fn constant(g: &Graph, v: TypedValue) -> Result<Node> {
    g.constant(v.t, v.value)
}

pub fn constant_scalar<T: TryInto<u128> + Not<Output = T> + TryInto<u8> + Copy>(
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
    g.ones(scalar_type(st))?.mixed_multiply(node)
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
        let pad = g.zeros(array_type(pad_shape, sc))?;
        g.concatenate(vec![data, pad], 0)?
    } else {
        data
    };
    let data = data.add(g.ones(scalar_type(BIT))?)?;
    let mut suffix_or = data;
    for i in 0..k {
        let shift = 2_i64.pow(i);
        suffix_or = g.multiply(
            suffix_or.get_slice(vec![SliceElement::SubArray(None, Some(-shift), None)])?,
            suffix_or.get_slice(vec![SliceElement::SubArray(Some(shift), None, None)])?,
        )?;
    }
    suffix_or.add(g.ones(scalar_type(BIT))?)
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

// Works only on positive integers from (0; 2^(2 * denominator_cap_2k)).
// First let's get the highest bit of the divisor.
// We group pairs of consecutive bits together for the purpose of the initial approximation.
// Namely, consider divisor to have digits (d_0, ..., d_31) in base-4. Then, if d_k is the highest
// non-zero digit, our approximation will be 2 ** (cap - k).
// Indeed, 4 ** k <= divisor < 4 ** (k + 1), so 2 ** (-k - 1) < 1 / sqrt(divisor) < 2 ** -k.
pub fn inverse_sqrt_initial_approximation(
    context: &Context,
    t: Type,
    denominator_cap_2k: u64,
) -> Result<Graph> {
    let sc = t.get_scalar_type();
    let g = context.create_graph()?;
    let divisor = g.input(t)?;
    let divisor_bits = pull_out_bits(divisor.a2b()?)?;
    let significant_bits = denominator_cap_2k * 2;
    let cum_or = cumulative_or(divisor_bits, significant_bits)?;
    let highest_one_bit_binary = g.add(
        cum_or.get_slice(vec![SliceElement::SubArray(
            None,
            Some(significant_bits as i64),
            None,
        )])?,
        cum_or.get_slice(vec![SliceElement::SubArray(
            Some(1),
            Some(significant_bits as i64 + 1),
            None,
        )])?,
    )?;
    let mut result = vec![];
    for i in 0..denominator_cap_2k {
        let index1 = 2 * denominator_cap_2k - 2 * i - 1;
        let index2 = 2 * denominator_cap_2k - 2 * i - 2;
        result.push(
            highest_one_bit_binary
                .get(vec![index1])?
                .add(highest_one_bit_binary.get(vec![index2])?)?,
        );
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

// One-hot encoding
// Inputs:
//   val: val is the value to be encoded. It has shape [arbitrary_shape, N] Where N is the number of bits after A2B.
//   max_val: max_val is the maximum value that can be encoded. max_val indicates the number of bits in the output.
//          The output will have shape [arbitrary_shape, max_val]. Obviously, max_val must be less than 2^N.
//   ids: ids is the precomputed array of values in the range (0..k) where k>=max_val. ids should be in binary format and have shape [k, N].
// Note that we assume that `val` is already A2B'ed.
// Output:
//  The output is a BIT array which is onehot encoding of val.
//  The output has shape [arbitrary_shape, max_val].
//
//  an efficient way to generate ids is to use the following code:
//  ```
//  let g = val.get_graph();
//  let ones = g.ones(array_type(vec![k],INT64))?;
//  let ids = ones.cum_sum(0)?.subtract(ones)?.a2b()?;
//  ```
pub fn onehot(val: Node, max_val: usize, ids: Node) -> Result<Node> {
    let ids = ids.get_slice(vec![SliceElement::SubArray(
        Some(0),
        Some(max_val as i64),
        None,
    )])?;
    let g = val.get_graph();
    // We need to calculate val == ids , but we need to compare bitwise
    // XOR(val_bits, Not(ids_bits)) is equivalent to val_bits == ids_bits
    // XOR is equivalent to addition modulo 2, and Not is equivalent to adding 1 modulo 2
    let bitwise_comparision = unsqueeze(val, -2)?.add(ids.add(constant_scalar(&g, 1, BIT)?)?)?;
    // val == ids only if bitwise_comparision is all ones
    // So we can reduce over the last dimension to get the result
    let res = reduce_mul(bitwise_comparision)?;
    Ok(res)
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

pub fn precise_goldschmidt_division(num: Node, den: Node) -> Result<Node> {
    let denominator_cap_2k = 30;
    let g = num.get_graph();
    let num = bits_64_to_128(num)?;
    let den = bits_64_to_128(den)?;
    let res = g.custom_op(
        CustomOperation::new(GoldschmidtDivision {
            iterations: 7,
            denominator_cap_2k,
        }),
        vec![num, den],
    )?;
    let res = bits_128_to_64(res, denominator_cap_2k)?;
    Ok(res)
}

fn u128_to_u64(a: Node, truncate_bits: i64) -> Result<Node> {
    a.a2b()?
        .get_slice(vec![
            SliceElement::Ellipsis,
            SliceElement::SubArray(Some(truncate_bits), Some(truncate_bits - 64), None),
        ])?
        .b2a(ScalarType::U64)
}

fn i128_to_i64(a: Node, truncate_bits: i64) -> Result<Node> {
    let g = a.get_graph();
    let a = a.a2b()?;
    let sgn = unsqueeze(
        a.get_slice(vec![SliceElement::Ellipsis, SliceElement::SingleIndex(-1)])?,
        -1,
    )?;
    // Truncate (remove least significant bits) and slice the rest.
    let a = a.get_slice(vec![
        SliceElement::Ellipsis,
        SliceElement::SubArray(Some(truncate_bits), Some(truncate_bits - 65), None),
    ])?;
    let last_axis = a.get_type()?.get_dimensions().len() - 1;
    // Concatenate sign bit with the rest of the bits.
    g.concatenate(vec![a, sgn], last_axis as u64)?
        .b2a(ScalarType::I64)
}

fn bits_128_to_64(a: Node, truncate_bits: u64) -> Result<Node> {
    let truncate_bits = truncate_bits as i64;
    if a.get_type()?.get_scalar_type().is_signed() {
        i128_to_i64(a, truncate_bits)
    } else {
        u128_to_u64(a, truncate_bits)
    }
}

fn u64_to_u128(a: Node) -> Result<Node> {
    // To convert u64 to u128 we need to append 64 0s to the most significant bits
    let g = a.get_graph();
    let a = a.a2b()?;
    let z = g.zeros(a.get_type()?)?;
    let last_axis = a.get_type()?.get_dimensions().len() - 1;
    g.concatenate(vec![a, z], last_axis as u64)?
        .b2a(ScalarType::U128)
}

fn i64_to_i128(a: Node) -> Result<Node> {
    let g = a.get_graph();
    let a = a.a2b()?;
    let t = a.get_type()?;
    let last_axis = t.get_dimensions().len() as u64 - 1;
    // If sign bit is 1, most significant bits after conversion will be 1s, otherwise 0s.
    let z = g.ones(a.get_type()?)?;

    // Extract sign bit.
    let sgn = unsqueeze(
        a.get_slice(vec![SliceElement::Ellipsis, SliceElement::SingleIndex(-1)])?,
        -1,
    )?;
    let z = z.multiply(sgn.clone())?;
    let a = a.get_slice(vec![
        SliceElement::Ellipsis,
        SliceElement::SubArray(None, Some(-1), None),
    ])?;
    // Concatenate sign bit to the end.
    g.concatenate(vec![a, z, sgn], last_axis)?
        .b2a(ScalarType::I128)
}

fn bits_64_to_128(a: Node) -> Result<Node> {
    if a.get_type()?.get_scalar_type().is_signed() {
        i64_to_i128(a)
    } else {
        u64_to_u128(a)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array};

    use super::*;
    use crate::{
        custom_ops::run_instantiation_pass,
        data_types::{scalar_type, INT64, UINT32},
        data_values::Value,
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
            let result = custom_reduce_helper(input, |first, second| first.add(second))?;
            let output = result.to_flattened_array_u64(array_type(vec![1], UINT32))?;
            assert_eq!(output, [(n * (n + 1) / 2) as u64]);
        }
        Ok(())
    }

    fn scalar_division_helper(dividend: i128, divisor: i128, st: ScalarType) -> Result<Value> {
        let c = simple_context(|g| {
            let dividend_node = g.input(scalar_type(st))?;
            let divisor_node = g.input(scalar_type(st))?;
            precise_goldschmidt_division(dividend_node, divisor_node)
        })?;
        let mapped_c = run_instantiation_pass(c)?;
        let result = random_evaluate(
            mapped_c.get_context().get_main_graph()?,
            vec![
                Value::from_scalar(dividend, st)?,
                Value::from_scalar(divisor, st)?,
            ],
        )?;
        Ok(result)
    }

    #[test]
    fn test_precise_goldschmidt_division() -> Result<()> {
        let dividend = 1234567890;
        let div_v = vec![1, 2, 3, 123, 300, 500, 700];
        for i in div_v {
            for st in [ScalarType::I64, ScalarType::U64] {
                let result = scalar_division_helper(dividend, i, st)?.to_i128(st)?;
                let expected = dividend / i;
                assert!(((result - expected).abs() * 100) / expected <= 1);
            }
        }
        let dividend = -1234567890;
        let div_v = vec![1, 2, 3, 123, 300, 500, 700];
        let st = ScalarType::I64;
        for i in div_v {
            let result = scalar_division_helper(dividend, i, st)?.to_i128(st)?;
            let expected = dividend / i;
            assert!(((result - expected).abs() * 100) / expected <= 1);
        }
        Ok(())
    }
}
