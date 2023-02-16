//! Binary adder that adds two bitstrings.
use crate::custom_ops::{CustomOperation, CustomOperationBody};
use crate::data_types::{array_type, Type, BIT};
use crate::errors::Result;
use crate::graphs::{Context, Graph, Node, SliceElement};
use crate::ops::utils::{constant_scalar, expand_dims, pull_out_bits, put_in_bits};

use serde::{Deserialize, Serialize};

use super::utils::validate_arguments_in_broadcast_bit_ops;

/// A structure that defines the custom operation BinaryAdd that implements the binary adder.
///
/// The binary adder takes two arrays of length-n bitstrings and returns the elementwise binary sum of these arrays, ignoring a possible overflow.
///
/// Only `n` which are powers of two are supported.
///
/// The last dimension of both inputs must be the same; it defines the length of input bitstrings.
/// If input shapes are different, the broadcasting rules are applied (see [the NumPy broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html)).
/// For example, if input arrays are of shapes `[2,3]`, and `[1,3]`, the resulting array has shape `[2,3]`.
///
/// Each bitstring of the output contains n bits; thus, this operation does not handle overflows.
///
/// This operation is needed for conversion between arithmetic and boolean additive MPC shares
/// (i.e. A2B and B2A operations in MPC).
///
/// To use this and other custom operations in computation graphs, see [Graph::custom_op].
///
/// # Custom operation arguments
///
/// - Node containing a binary array or scalar
/// - Node containing a binary array or scalar
///
/// # Custom operation returns
///
/// New BinaryAdd node
///
/// # Example
///
/// ```
/// # use ciphercore_base::graphs::create_context;
/// # use ciphercore_base::data_types::{array_type, BIT};
/// # use ciphercore_base::custom_ops::{CustomOperation};
/// # use ciphercore_base::ops::adder::BinaryAdd;
/// let c = create_context().unwrap();
/// let g = c.create_graph().unwrap();
/// let t = array_type(vec![2, 4], BIT);
/// let n1 = g.input(t.clone()).unwrap();
/// let n2 = g.input(t.clone()).unwrap();
/// let n3 = g.custom_op(CustomOperation::new(BinaryAdd {}), vec![n1, n2]).unwrap();
/// ```
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct BinaryAdd {}

#[typetag::serde]
impl CustomOperationBody for BinaryAdd {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        validate_arguments_in_broadcast_bit_ops(arguments_types.clone(), &self.get_name())?;
        let input_type0 = arguments_types[0].clone();
        let input_type1 = arguments_types[1].clone();

        // Adder input consists of two binary strings x and y
        let g = context.create_graph()?;
        let input0 = pull_out_bits(g.input(input_type0)?)?;
        let input1 = pull_out_bits(g.input(input_type1)?)?;
        let added = g.custom_op(
            CustomOperation::new(BinaryAddTransposed {}),
            vec![input0, input1],
        )?;
        let output = put_in_bits(added)?;
        output.set_as_output()?;
        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        "BinaryAdd".to_owned()
    }
}

// Same as BinaryAdd, but expect that the first dimension is bits.
// This is a performance optimization, it's easier to operate on the first dimension.
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub(crate) struct BinaryAddTransposed {}

#[typetag::serde]
impl CustomOperationBody for BinaryAddTransposed {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        if arguments_types.len() != 2 {
            return Err(runtime_error!("Invalid number of arguments"));
        }
        match (&arguments_types[0], &arguments_types[1]) {
            (Type::Array(shape0, scalar_type0), Type::Array(shape1, scalar_type1)) => {
                if shape0[0] != shape1[0] {
                    return Err(runtime_error!(
                        "Input arrays' first dimensions are not the same"
                    ));
                }
                if *scalar_type0 != BIT {
                    return Err(runtime_error!("Input array [0]'s ScalarType is not BIT"));
                }
                if *scalar_type1 != BIT {
                    return Err(runtime_error!("Input array [1]'s ScalarType is not BIT"));
                }
            }
            _ => {
                return Err(runtime_error!(
                    "Invalid input argument type, expected Array type"
                ));
            }
        }

        let input_type0 = arguments_types[0].clone();
        let input_type1 = arguments_types[1].clone();

        // Adder input consists of two binary strings x and y
        let g = context.create_graph()?;
        let input0 = g.input(input_type0)?;
        let input1 = g.input(input_type1)?;
        // Compute "propagate" bits x_i XOR y_i
        let xor_bits = g.add(input0.clone(), input1.clone())?;
        // Compute "generate" bits x_i AND y_i
        let and_bits = g.multiply(input0, input1)?;

        let carries = calculate_carry_bits(xor_bits.clone(), and_bits)?;
        // The last step is to XOR carries with "propagate" bits
        let output = carries.add(xor_bits)?;
        output.set_as_output()?;
        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        "BinaryAddTransposed".to_owned()
    }
}

/// Actual carry is calculated as `generate + propagate * prev_carry`
///
/// `(propagate, generate) = (1, 1)` is impossible state.
#[derive(Clone)]
struct CarryNode {
    propagate: Node,
    generate: Node,
}

impl CarryNode {
    fn bit_len(&self) -> Result<u64> {
        Ok(self.propagate.get_type()?.get_shape()[0])
    }

    fn shrink(&self) -> Result<CarryNode> {
        let bit_len = self.bit_len()? as i64;

        let next_lvl_bits = (bit_len - 1) / 2;
        let use_bits = next_lvl_bits * 2;
        let lower = self.sub_slice(0, use_bits)?;
        let higher = self.sub_slice(1, use_bits)?;

        lower.join(&higher)
    }

    /// assumes `bit_len` is the same for `self` and `rhs`
    fn join(&self, rhs: &Self) -> Result<Self> {
        let propagate = self.propagate.multiply(rhs.propagate.clone())?;
        let generate = rhs
            .generate
            .add(rhs.propagate.multiply(self.generate.clone())?)?;
        Ok(Self {
            propagate,
            generate,
        })
    }

    /// Returns every second element starting from `start_offset`
    fn sub_slice(&self, start_offset: i64, bit_len: i64) -> Result<Self> {
        let get_slice = |node: &Node| {
            node.get_slice(vec![SliceElement::SubArray(
                Some(start_offset),
                Some(bit_len),
                Some(2),
            )])
        };
        Ok(Self {
            propagate: get_slice(&self.propagate)?,
            generate: get_slice(&self.generate)?,
        })
    }

    fn apply(&self, prev_carry: Node) -> Result<Node> {
        self.generate.add(self.propagate.multiply(prev_carry)?)
    }
}

/// Takes arrays `[a1, a2, ..., a_n]` and `[b1, b2, ..., b_n]`
///
/// Returns `[a1, b1, a2, b2, ..., a_n, b_n]`
fn interleave(first: Node, second: Node) -> Result<Node> {
    let first = expand_dims(first, &[0])?;
    let second = expand_dims(second, &[0])?;
    let graph = first.get_graph();
    let joined = graph.concatenate(vec![first, second], 0)?;
    let mut axes: Vec<_> = (0..joined.get_type()?.get_shape().len() as u64).collect();
    axes.swap(0, 1);
    let joined = joined.permute_axes(axes)?;
    let mut shape = joined.get_type()?.get_shape();
    shape[0] *= 2;
    shape.remove(1);
    let scalar = joined.get_type()?.get_scalar_type();
    joined.reshape(array_type(shape, scalar))
}

/// This function generates a graph for the "segment tree" to calculate
/// carry bits.
///
/// It assumes both `propagate_bits` and `generate_bits` are arrays
/// with bits dimension pulled out to the outermost level.
/// It also assumes the number of bits is a power of two.
///
/// Each node of the segment tree stores two bits `(propagate, generate)`.
/// Based on the carry bit from the previous node, a new carry bit could be calculated
/// as `generate + propagate * prev_carry`.
///
/// The overall multiplicative depth of generated segment tree is `2*log(bits)`.
/// First, we generate nodes for each separate bit.
/// Then, we join neighboring nodes, until we have at most two nodes. We don't need
/// to do the last join as we don't care about overflows.
///
/// When the top node is calculated, we go top-down and push carry bits to the lower
/// levels.
///
/// # Example
/// Let's say we have 8 bits. First, we create a node for each bit:
/// ```text
/// | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |    <- stored in `nodes[0]`
///   \   /   \   /   \   /
///    01      23      45                <- stored in `nodes[1]`
///      \     /
///        03                            <- stored in `nodes[2]`
/// ```
///
/// Then we calculate carry bits based on nodes from top to bottom. We know
/// that `carry[0] = 0`.
///
/// Based on `nodes[2]` we calculate `carry[4]`, interleave with `carry[0]`,
/// and get `{carry[0], carry[4]}`.
///
/// Then we use `nodes[1]` to generate `{carry[2], carry[6]}`, interleave with
/// previous result, and get `{carry[0], carry[2], carry[4], carry[6]}`. Note
/// that to do this we only need half of the values from `nodes[1]` (we don't need node `23`).
///
/// And finally using (half of) `nodes[0]` we can calculate all odd indexes carries.
///
fn calculate_carry_bits(propagate_bits: Node, generate_bits: Node) -> Result<Node> {
    let graph = propagate_bits.get_graph();

    let mut nodes = vec![CarryNode {
        propagate: propagate_bits,
        generate: generate_bits,
    }];
    if !nodes[0].bit_len()?.is_power_of_two() {
        return Err(runtime_error!("BinaryAdd only supports numbers with number of bits, which is a power of 2. {} bits provided.", nodes[0].bit_len()?));
    }
    while nodes.last().unwrap().bit_len()? > 2 {
        let last = nodes.last().unwrap();
        nodes.push(last.shrink()?);
    }

    let zero = constant_scalar(&graph, 0, BIT)?;
    let mut carries = nodes[0]
        .propagate
        .get_slice(vec![SliceElement::SubArray(Some(0), Some(1), None)])?
        .multiply(zero)?;
    for node in nodes.iter().rev() {
        let lower = node.sub_slice(0, node.bit_len()? as i64)?;
        let new_carries = lower.apply(carries.clone())?;
        carries = interleave(carries, new_carries)?;
    }

    Ok(carries)
}

#[cfg(test)]
mod tests {
    use std::ops::Not;

    use super::*;

    use crate::custom_ops::{run_instantiation_pass, CustomOperation};
    use crate::data_types::{array_type, create_scalar_type, tuple_type, INT16, INT64};
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::graphs::create_context;
    use crate::graphs::util::simple_context;

    fn test_helper(first: u64, second: u64, bits: u64) -> Result<()> {
        let modulus = if bits == 64 {
            None
        } else {
            Some(2u64.pow(bits as u32))
        };
        let mask = if let Some(modulus) = modulus {
            modulus - 1
        } else {
            0u64.not()
        };
        let first = first & mask;
        let second = second & mask;

        let c = simple_context(|g| {
            let i1 = g.input(array_type(vec![bits], BIT))?;
            let i2 = g.input(array_type(vec![bits], BIT))?;
            g.custom_op(CustomOperation::new(BinaryAdd {}), vec![i1, i2])
        })?;
        let mapped_c = run_instantiation_pass(c)?;
        let scalar = create_scalar_type(false, modulus);
        let input0 = Value::from_scalar(first, scalar.clone())?;
        let input1 = Value::from_scalar(second, scalar.clone())?;
        let result_v = random_evaluate(
            mapped_c.get_context().get_main_graph()?,
            vec![input0, input1],
        )?
        .to_u64(scalar.clone())?;

        let expected_result = first.wrapping_add(second) & mask;
        assert_eq!(result_v, expected_result);
        Ok(())
    }

    #[test]
    fn test_random_inputs() -> Result<()> {
        let random_numbers = [0, 1, 3, 4, 10, 100500, 123456, 787788];
        for bits in (0..=6).map(|pw| 2u64.pow(pw)) {
            for &x in random_numbers.iter() {
                for &y in random_numbers.iter() {
                    test_helper(x, y, bits)?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_well_behaved() -> Result<()> {
        {
            let c = simple_context(|g| {
                let i1 = g.input(array_type(vec![5, 16], BIT))?;
                let i2 = g.input(array_type(vec![1, 16], BIT))?;
                g.custom_op(CustomOperation::new(BinaryAdd {}), vec![i1, i2])
            })?;
            let mapped_c = run_instantiation_pass(c)?;
            let inputs1 =
                Value::from_flattened_array(&vec![0, 1023, -1023, i16::MIN, i16::MAX], INT16)?;
            let inputs2 = Value::from_flattened_array(&vec![1024], INT16)?;
            let result_v = random_evaluate(
                mapped_c.get_context().get_main_graph()?,
                vec![inputs1, inputs2],
            )?
            .to_flattened_array_u64(array_type(vec![5], INT16))?;
            assert_eq!(
                result_v,
                vec![1024, 2047, 1, (1u64 << 15) + 1024, (1u64 << 15) - 1 + 1024]
            );
        }
        {
            let c = simple_context(|g| {
                let i1 = g.input(array_type(vec![64], BIT))?;
                let i2 = g.input(array_type(vec![64], BIT))?;
                g.custom_op(CustomOperation::new(BinaryAdd {}), vec![i1, i2])
            })?;
            let mapped_c = run_instantiation_pass(c)?;
            let input0 = Value::from_scalar(123456790, INT64)?;
            let input1 = Value::from_scalar(-123456789, INT64)?;
            let result_v = random_evaluate(
                mapped_c.get_context().get_main_graph()?,
                vec![input0, input1],
            )?
            .to_u64(INT64)?;
            assert_eq!(result_v, 1);
        }
        Ok(())
    }

    #[test]
    fn test_malformed() -> Result<()> {
        let c = create_context()?;
        let g = c.create_graph()?;
        let i = g.input(array_type(vec![64], BIT))?;
        let i1 = g.input(array_type(vec![64], INT16))?;
        let i2 = g.input(tuple_type(vec![]))?;
        let i3 = g.input(array_type(vec![32], BIT))?;
        let i4 = g.input(array_type(vec![31], BIT))?;
        assert!(g
            .custom_op(CustomOperation::new(BinaryAdd {}), vec![i.clone()])
            .is_err());
        assert!(g
            .custom_op(
                CustomOperation::new(BinaryAdd {}),
                vec![i.clone(), i1.clone()]
            )
            .is_err());
        assert!(g
            .custom_op(
                CustomOperation::new(BinaryAdd {}),
                vec![i1.clone(), i.clone()]
            )
            .is_err());
        assert!(g
            .custom_op(CustomOperation::new(BinaryAdd {}), vec![i2])
            .is_err());
        assert!(g
            .custom_op(CustomOperation::new(BinaryAdd {}), vec![i.clone(), i3])
            .is_err());
        assert!(g
            .custom_op(CustomOperation::new(BinaryAdd {}), vec![i4.clone(), i4])
            .is_err());
        Ok(())
    }
}
