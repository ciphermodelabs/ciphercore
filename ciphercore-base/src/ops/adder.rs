use crate::broadcast::broadcast_shapes;
use crate::custom_ops::CustomOperationBody;
use crate::data_types::{array_type, scalar_type, tuple_type, Type, BIT};
use crate::data_values::Value;
use crate::errors::Result;
use crate::graphs::{Context, Graph, GraphAnnotation};
use crate::ops::utils::{pull_out_bits, put_in_bits, validate_arguments_in_broadcast_bit_ops};

use serde::{Deserialize, Serialize};

/// A structure that defines the custom operation BinaryAdd that implements the binary adder.
///
/// The binary adder takes two arrays of length-n bitstrings and returns the elementwise binary sum of these arrays, ignoring a possible overflow.
///
/// The last dimension of both inputs must be the same; it defines the length of input bitstrings.
/// If input shapes are different, the broadcasting rules are applied (see [the NumPy broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html)).
/// For example, if input arrays are of shapes `[2,3]`, and `[1,3]`, the resulting array has shape `[2,3]`.
///
/// Each bitstring of the output contains n bits; thus, this operation does not handle overflows.
///
/// The binary adder is either depth- or size-optimized depending on [the inlining mode](crate::inline::inline_ops::InlineMode).
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
/// let t = array_type(vec![2, 3], BIT);
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

        let input_shape0 = input_type0.get_shape();
        let input_shape1 = input_type1.get_shape();

        let output_shape = broadcast_shapes(input_shape0, input_shape1)?;

        let output_type = if output_shape.len() == 1 {
            scalar_type(BIT)
        } else {
            array_type(output_shape[0..output_shape.len() - 1].to_vec(), BIT)
        };

        // Prefix sum graph computing carry bits.
        // Its input is a tuple of "propagate" and "generate" bits.
        // The state contains the carry computed in the previous iteration.
        // Its output is equal to the previous carry.
        // The resulting state is equal to the carry of the current iteration = and_bits + xor_bits * previous carry.
        let ps_g = context.create_graph()?;
        {
            let state = ps_g.input(output_type.clone())?;
            let xor_and_bits =
                ps_g.input(tuple_type(vec![output_type.clone(), output_type.clone()]))?;
            let output_state = state
                .multiply(xor_and_bits.tuple_get(0)?)?
                .add(xor_and_bits.tuple_get(1)?)?;
            let output = ps_g.create_tuple(vec![output_state, state])?;
            output.set_as_output()?;
            ps_g.add_annotation(GraphAnnotation::OneBitState)?;
            ps_g.finalize()?;
        }

        // Adder input consists of two binary strings x and y
        let g = context.create_graph()?;
        let input0 = g.input(input_type0)?;
        let input1 = g.input(input_type1)?;
        // Compute "propagate" bits x_i XOR y_i
        let xor_bits = g.add(input0.clone(), input1.clone())?;
        // Compute "generate" bits x_i AND y_i
        let and_bits = g.multiply(input0, input1)?;

        let pulled_out_xor_bits = pull_out_bits(xor_bits.clone())?.array_to_vector()?;
        let pulled_out_and_bits = pull_out_bits(and_bits)?.array_to_vector()?;
        let zip_xor_and = g.zip(vec![pulled_out_xor_bits, pulled_out_and_bits])?;

        let zero_bit = g.constant(output_type.clone(), Value::zero_of_type(output_type))?;
        let pulled_out_carries_vec = g.iterate(ps_g, zero_bit, zip_xor_and)?.tuple_get(1)?;
        let pulled_out_carries = pulled_out_carries_vec.vector_to_array()?;
        let carries = put_in_bits(pulled_out_carries)?;
        // The last step is to XOR carries with "propagate" bits
        let output = carries.add(xor_bits)?;
        output.set_as_output()?;
        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        "BinaryAdd".to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::custom_ops::{run_instantiation_pass, CustomOperation};
    use crate::data_types::{array_type, INT16, INT64};
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::graphs::create_context;

    #[test]
    fn test_well_behaved() {
        || -> Result<()> {
            let c = create_context()?;
            let g = c.create_graph()?;
            let i1 = g.input(array_type(vec![5, 16], BIT))?;
            let i2 = g.input(array_type(vec![1, 16], BIT))?;
            let o = g.custom_op(CustomOperation::new(BinaryAdd {}), vec![i1, i2])?;
            g.set_output_node(o)?;
            g.finalize()?;
            c.set_main_graph(g)?;
            c.finalize()?;
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
            Ok(())
        }()
        .unwrap();
        || -> Result<()> {
            let c = create_context()?;
            let g = c.create_graph()?;
            let i1 = g.input(array_type(vec![64], BIT))?;
            let i2 = g.input(array_type(vec![64], BIT))?;
            let o = g.custom_op(CustomOperation::new(BinaryAdd {}), vec![i1, i2])?;
            g.set_output_node(o)?;
            g.finalize()?;
            c.set_main_graph(g)?;
            c.finalize()?;
            let mapped_c = run_instantiation_pass(c)?;
            let input0 = Value::from_scalar(123456790, INT64)?;
            let input1 = Value::from_scalar(-123456789, INT64)?;
            let result_v = random_evaluate(
                mapped_c.get_context().get_main_graph()?,
                vec![input0, input1],
            )?
            .to_u64(INT64)?;
            assert_eq!(result_v, 1);
            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_malformed() {
        || -> Result<()> {
            let c = create_context()?;
            let g = c.create_graph()?;
            let i = g.input(array_type(vec![64], BIT))?;
            let i1 = g.input(array_type(vec![64], INT16))?;
            let i2 = g.input(tuple_type(vec![]))?;
            let i3 = g.input(array_type(vec![32], BIT))?;
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
            Ok(())
        }()
        .unwrap();
    }
}
