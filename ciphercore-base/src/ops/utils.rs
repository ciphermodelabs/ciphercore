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
    let dims = match node.get_type()? {
        Type::Array(shape, _) => shape,
        _ => return Err(runtime_error!("Expected array")),
    };
    let mut dim = dims[dims.len() - 1];
    let mut trans_node = pull_out_bits(node)?;
    let mut stray_bits = vec![];
    while dim > 1 {
        if dim % 2 == 1 {
            stray_bits.push(trans_node.get(vec![0])?);
            trans_node = trans_node.get_slice(vec![SliceElement::SubArray(Some(1), None, None)])?;
            dim -= 1;
        } else {
            let half1 = trans_node.get_slice(vec![SliceElement::SubArray(
                Some(0),
                Some((dim / 2) as i64),
                None,
            )])?;
            let half2 = trans_node.get_slice(vec![SliceElement::SubArray(
                Some((dim / 2) as i64),
                None,
                None,
            )])?;
            trans_node = half1.multiply(half2)?;
            dim /= 2;
        }
    }
    trans_node = trans_node.get(vec![0])?;
    for bit in stray_bits {
        trans_node = trans_node.multiply(bit)?;
    }
    Ok(trans_node)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        custom_ops::run_instantiation_pass,
        data_types::{scalar_type, INT64},
        evaluators::random_evaluate,
        graphs::create_context,
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
}
