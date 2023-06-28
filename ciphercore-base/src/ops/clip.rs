//! Clip function that returns a given value if it is inside of the interval [0,2<sup>k</sup>] and clips values outside this interval to its edges.
use crate::custom_ops::{CustomOperation, CustomOperationBody, Or};
use crate::data_types::{array_type, scalar_type, vector_type, Type, BIT};
use crate::errors::Result;
use crate::graphs::{Context, Graph, GraphAnnotation, SliceElement};
use crate::ops::multiplexer::Mux;
use crate::ops::utils::{pull_out_bits, put_in_bits};

use serde::{Deserialize, Serialize};

/// A structure that defines the custom operation Clip2K that computes elementwise the following clipping function:
/// - 0 if input <= 0,
/// - input if 0 < input < 2<sup>k</sup>,
/// - 2<sup>k</sup> if >= 2<sup>k</sup>.
///
/// This function is an approximation of [the sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function).
///
/// An array of length-n bitstrings is accepted as input. These bitstrings are interpreted as signed integers.
///
/// To use this and other custom operations in computation graphs, see [Graph::custom_op].
///
/// # Custom operation arguments
///
/// - Node containing a binary array
///
/// # Custom operation returns
///
/// New Clip2K node
///
/// # Example
///
/// ```
/// # use ciphercore_base::graphs::create_context;
/// # use ciphercore_base::data_types::{array_type, BIT};
/// # use ciphercore_base::custom_ops::{CustomOperation};
/// # use ciphercore_base::ops::clip::Clip2K;
/// let c = create_context().unwrap();
/// let g = c.create_graph().unwrap();
/// let t = array_type(vec![2, 16], BIT);
/// let n1 = g.input(t.clone()).unwrap();
/// let n2 = g.custom_op(CustomOperation::new(Clip2K {k: 4}), vec![n1]).unwrap();
/// ```
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct Clip2K {
    /// 2<sup>k</sup> is the upper threshold of clipping
    pub k: u64,
}

#[typetag::serde]
impl CustomOperationBody for Clip2K {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        if arguments_types.len() != 1 {
            return Err(runtime_error!("Invalid number of arguments for Clip"));
        }
        let input_type = arguments_types[0].clone();
        if !input_type.is_array() || input_type.get_scalar_type() != BIT {
            return Err(runtime_error!("Clip can only be applied to bit arrays"));
        }
        let input_shape = input_type.get_shape();
        let num_bits = input_shape[input_shape.len() - 1];
        if self.k >= num_bits - 1 {
            return Err(runtime_error!(
                "Clip(k) can be applied only whenever k <= num_bits - 2"
            ));
        }
        let bit_type = if input_shape.len() == 1 {
            scalar_type(BIT)
        } else {
            array_type(input_shape[0..input_shape.len() - 1].to_vec(), BIT)
        };
        let aux_or_graph = context.create_graph()?;
        let state = aux_or_graph.input(bit_type.clone())?;
        let input = aux_or_graph.input(bit_type.clone())?;
        let output_state =
            aux_or_graph.custom_op(CustomOperation::new(Or {}), vec![state, input])?;
        let empty = aux_or_graph.create_tuple(vec![])?;
        let output = aux_or_graph.create_tuple(vec![output_state, empty])?;
        aux_or_graph.set_output_node(output)?;
        aux_or_graph.add_annotation(GraphAnnotation::AssociativeOperation)?;
        aux_or_graph.finalize()?;
        let g = context.create_graph()?;
        let input = g.input(input_type)?;
        let input_bits = pull_out_bits(input)?;
        let is_negative = input_bits.get(vec![num_bits - 1])?;
        let zero_bit = g.zeros(bit_type.clone())?;
        let one_bit = g.ones(bit_type.clone())?;
        let top_bits = input_bits
            .get_slice(vec![SliceElement::SubArray(
                Some(self.k as i64),
                None,
                None,
            )])?
            .array_to_vector()?;
        let is_large_or_negative = g
            .iterate(aux_or_graph, zero_bit.clone(), top_bits)?
            .tuple_get(0)?;
        // clipped_value = if is_negative then 0, else 2^k
        // obtained by concatenating a bunch of zeros,
        // zero or one, then bunch of zeros again
        let clipped_value = g
            .create_tuple(vec![
                zero_bit.repeat(self.k)?,
                g.custom_op(
                    CustomOperation::new(Mux {}),
                    vec![is_negative, zero_bit.clone(), one_bit],
                )?,
                zero_bit.repeat(num_bits - self.k - 1)?,
            ])?
            .reshape(vector_type(num_bits, bit_type))?
            .vector_to_array()?;
        g.set_output_node(put_in_bits(g.custom_op(
            CustomOperation::new(Mux {}),
            vec![is_large_or_negative, clipped_value, input_bits],
        )?)?)?;
        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        format!("Clip({})", self.k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::custom_ops::{run_instantiation_pass, CustomOperation};
    use crate::data_types::{array_type, tuple_type, INT32, INT64};
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::graphs::create_context;
    use crate::graphs::util::simple_context;

    #[test]
    fn test_well_behaved() {
        || -> Result<()> {
            let c = simple_context(|g| {
                let i = g.input(array_type(vec![19, 64], BIT))?;
                g.custom_op(CustomOperation::new(Clip2K { k: 10 }), vec![i])
            })?;
            let mapped_c = run_instantiation_pass(c)?;
            let inputs = Value::from_flattened_array(
                &[
                    0,
                    1,
                    -1,
                    2,
                    -2,
                    1023,
                    -1023,
                    1024,
                    -1024,
                    1025,
                    -1025,
                    2048,
                    -2048,
                    2047,
                    -2047,
                    2049,
                    -2049,
                    i64::MIN,
                    i64::MAX,
                ],
                INT64,
            )?;
            let result_v = random_evaluate(mapped_c.get_context().get_main_graph()?, vec![inputs])?
                .to_flattened_array_u64(array_type(vec![19], INT64))?;
            assert_eq!(
                result_v,
                vec![0, 1, 0, 2, 0, 1023, 0, 1024, 0, 1024, 0, 1024, 0, 1024, 0, 1024, 0, 0, 1024]
            );
            Ok(())
        }()
        .unwrap();
        || -> Result<()> {
            let c = simple_context(|g| {
                let i = g.input(array_type(vec![64], BIT))?;
                g.custom_op(CustomOperation::new(Clip2K { k: 20 }), vec![i])
            })?;
            let mapped_c = run_instantiation_pass(c)?;
            let inputs = Value::from_scalar(123456789, INT64)?;
            let result_v = random_evaluate(mapped_c.get_context().get_main_graph()?, vec![inputs])?
                .to_u64(INT64)?;
            assert_eq!(result_v, 1 << 20);
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
            let i1 = g.input(array_type(vec![64], INT32))?;
            let i2 = g.input(tuple_type(vec![]))?;
            assert!(g
                .custom_op(CustomOperation::new(Clip2K { k: 64 }), vec![i])
                .is_err());
            assert!(g
                .custom_op(CustomOperation::new(Clip2K { k: 20 }), vec![])
                .is_err());
            assert!(g
                .custom_op(CustomOperation::new(Clip2K { k: 20 }), vec![i1])
                .is_err());
            assert!(g
                .custom_op(CustomOperation::new(Clip2K { k: 20 }), vec![i2])
                .is_err());
            Ok(())
        }()
        .unwrap();
    }
}
