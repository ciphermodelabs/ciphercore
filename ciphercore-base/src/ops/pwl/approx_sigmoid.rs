//! Sigmoid(x) piecewise-linear approximation.
use crate::custom_ops::CustomOperationBody;
use crate::data_types::{Type, INT64};
use crate::errors::Result;
use crate::graphs::{Context, Graph};

use serde::{Deserialize, Serialize};

use super::approx_pointwise::{create_approximation, PWLConfig};

/// A structure that defines the custom operation ApproxSigmoid that computes an approximate Sigmoid(x / (2 ** precision)) * (2 ** precision) using piecewise-linear approximation.
///
/// Sigmoid is a very commonly used function in ML: Sigmoid(x) = 1 / (1 + exp(-x)).
/// So far this operation supports only INT64 scalar type.
///
/// # Custom operation arguments
///
/// - Node containing a signed 64-bit array or scalar to compute the sigmoid
///
/// # Custom operation returns
///
/// New ApproxSigmoid node
///
/// # Example
///
/// ```
/// # use ciphercore_base::graphs::create_context;
/// # use ciphercore_base::data_types::{scalar_type, array_type, INT64};
/// # use ciphercore_base::custom_ops::{CustomOperation};
/// # use ciphercore_base::ops::pwl::approx_sigmoid::ApproxSigmoid;
/// let c = create_context().unwrap();
/// let g = c.create_graph().unwrap();
/// let t = array_type(vec![3], INT64);
/// let x = g.input(t.clone()).unwrap();
/// let n = g.custom_op(CustomOperation::new(ApproxSigmoid {precision: 4}), vec![x]).unwrap();
///
// TODO: generalize to other types.
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct ApproxSigmoid {
    /// Assume that we're operating in fixed precision arithmetic with denominator 2 ** precision.
    pub precision: u64,
}

#[typetag::serde]
impl CustomOperationBody for ApproxSigmoid {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        if arguments_types.len() != 1 {
            return Err(runtime_error!(
                "Invalid number of arguments for ApproxSigmoid"
            ));
        }
        let t = arguments_types[0].clone();
        if !t.is_scalar() && !t.is_array() {
            return Err(runtime_error!(
                "Argument in ApproxSigmoid must be a scalar or an array"
            ));
        }
        let sc = t.get_scalar_type();
        if sc != INT64 {
            return Err(runtime_error!(
                "Argument in ApproxSigmoid must consist of INT64's"
            ));
        }
        if self.precision > 30 || self.precision == 0 {
            return Err(runtime_error!("`precision` should be in range [1, 30]."));
        }

        let g = context.create_graph()?;
        let arg = g.input(t)?;
        // Choice of parameters:
        // -- left/right: our typical use-case is precision=15, leading to minimum value around 3e-5. Sigmoid(-10) is 4.5e-5, so right=-left=10 is a reasonable choice with our precision;
        // -- log_buckets: we look at max absolute difference to the real sigmoid. It looks as follows:
        //    log_buckets=4 => 0.0163,
        //    log_buckets=5 => 0.0045,
        //    log_buckets=6 => 0.0012.
        // After 5 segments, we're getting diminishing returns, so it doesn't make sense to go higher (for the sake of performance).
        // -- flatten_left/flatten_right: sigmoid is flat on both sides.
        let result = create_approximation(
            arg,
            |x| 1.0 / (1.0 + (-x).exp()),
            -10.0,
            10.0,
            self.precision,
            PWLConfig {
                log_buckets: 5,
                flatten_left: true,
                flatten_right: true,
            },
        )?;
        result.set_as_output()?;
        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        format!("ApproxSigmoid(scaling_factor=2**{})", self.precision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::custom_ops::run_instantiation_pass;
    use crate::custom_ops::CustomOperation;
    use crate::data_types::array_type;
    use crate::data_types::scalar_type;
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::graphs::util::simple_context;

    fn scalar_helper(arg: i64, precision: u64) -> Result<i64> {
        let c = simple_context(|g| {
            let i = g.input(scalar_type(INT64))?;
            g.custom_op(CustomOperation::new(ApproxSigmoid { precision }), vec![i])
        })?;
        let mapped_c = run_instantiation_pass(c)?;
        let result = random_evaluate(
            mapped_c.get_context().get_main_graph()?,
            vec![Value::from_scalar(arg, INT64)?],
        )?;
        let res = result.to_i64(INT64)?;
        Ok(res)
    }

    fn array_helper(arg: Vec<i64>) -> Result<Vec<i64>> {
        let array_t = array_type(vec![arg.len() as u64], INT64);
        let c = simple_context(|g| {
            let i = g.input(array_t.clone())?;
            g.custom_op(
                CustomOperation::new(ApproxSigmoid { precision: 10 }),
                vec![i],
            )
        })?;
        let mapped_c = run_instantiation_pass(c)?;
        let result = random_evaluate(
            mapped_c.get_context().get_main_graph()?,
            vec![Value::from_flattened_array(&arg, INT64)?],
        )?;
        result.to_flattened_array_i64(array_t)
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    #[test]
    fn test_approx_sigmoid_scalar() {
        for i in (-5000..5000).step_by(1000) {
            let expected = (sigmoid((i as f32) / 1024.0) * 1024.0) as i64;
            let actual = scalar_helper(i, 10).unwrap();
            let absolute_error = ((expected - actual).abs() as f64) / 1024.0;
            assert!(absolute_error <= 0.01);
        }
    }

    #[test]
    fn test_approx_sigmoid_array() {
        let arr: Vec<i64> = (-5000..5000).step_by(100).collect();
        let res = array_helper(arr.clone()).unwrap();
        for i in 0..arr.len() {
            let expected = (sigmoid((arr[i] as f32) / 1024.0) * 1024.0) as i64;
            let actual = res[i];
            let absolute_error = ((expected - actual).abs() as f64) / 1024.0;
            assert!(absolute_error <= 0.01);
        }
    }
}
