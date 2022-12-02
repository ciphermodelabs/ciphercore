use std::ops::Not;

use crate::data_types::{array_type, scalar_type, ScalarType, Type, BIT, UINT64};
use crate::data_values::Value;
use crate::errors::Result;
use crate::graphs::{Context, Graph, GraphAnnotation, Node};
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
    let mut error_message: String = format!("{}: ", custom_op_name);
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
    let ones = if node.get_type()?.is_array() {
        zeros(
            &node.get_graph(),
            array_type(node.get_type()?.get_shape(), st.clone()),
        )?
    } else {
        zeros(&node.get_graph(), scalar_type(st.clone()))?
    }
    .add(constant_scalar(&node.get_graph(), 1, st)?)?;
    ones.mixed_multiply(node)
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

// Another apporach to approximate the inital value is proposed here:
// https://www.ifca.ai/pub/fc10/31_47.pdf section 3.4
// However, the current approach seems to work fine for Goldschmidt's as well as Newton's method.
pub fn inverse_initial_approximation(
    context: &Context,
    t: Type,
    denominator_cap_2k: u64,
) -> Result<Graph> {
    let sc = t.get_scalar_type();

    let bit_type = if t.is_scalar() {
        scalar_type(BIT)
    } else {
        array_type(t.get_shape(), BIT)
    };
    // Graph for identifying highest one bit.
    let g_highest_one_bit = context.create_graph()?;
    {
        let input_state = g_highest_one_bit.input(bit_type.clone())?;
        let input_bit = g_highest_one_bit.input(bit_type.clone())?;

        let one = constant_scalar(&g_highest_one_bit, 1, BIT)?;
        let not_input_state = one.add(input_state.clone())?;
        // If input state is 1, then the highest bit has been already encountered.
        // All other bits can be set to zero.
        let output = not_input_state.multiply(input_bit)?;
        // new_state is equal to input_state OR input_bit
        // Hence, input state becomes and stays 1 once the highest bit has been encountered.
        let new_state = input_state.add(output.clone())?;
        let output_tuple = g_highest_one_bit.create_tuple(vec![new_state, output])?;
        output_tuple.set_as_output()?;
    }
    g_highest_one_bit.add_annotation(GraphAnnotation::AssociativeOperation)?;
    g_highest_one_bit.finalize()?;

    let g = context.create_graph()?;
    let divisor = g.input(t)?;
    let divisor_bits = pull_out_bits(divisor.a2b()?)?.array_to_vector()?;
    let mut divisor_bits_reversed = vec![];
    for i in 0..denominator_cap_2k {
        let index = constant_scalar(&g, denominator_cap_2k - i - 1, UINT64)?;
        divisor_bits_reversed.push(divisor_bits.vector_get(index)?);
    }
    let zero = zeros(&g, bit_type.clone())?;
    let highest_one_bit_binary = g
        .iterate(
            g_highest_one_bit,
            zero,
            g.create_vector(bit_type, divisor_bits_reversed)?,
        )?
        .tuple_get(1)?
        .vector_to_array()?;
    let highest_one_bit = single_bit_to_arithmetic(highest_one_bit_binary, sc.clone())?;
    let first_approximation_bits = put_in_bits(highest_one_bit)?;
    let mut powers_of_two = vec![];
    for i in 0..denominator_cap_2k {
        powers_of_two.push(1u64 << i);
    }
    let powers_of_two_node = g.constant(
        array_type(vec![denominator_cap_2k], sc.clone()),
        Value::from_flattened_array(&powers_of_two, sc)?,
    )?;
    let approximation = first_approximation_bits.dot(powers_of_two_node)?;
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
