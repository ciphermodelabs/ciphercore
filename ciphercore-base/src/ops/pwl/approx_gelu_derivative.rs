//! Derivative of GELU(x) piecewise-linear approximation.
use crate::custom_ops::CustomOperationBody;
use crate::data_types::{Type, INT64};
use crate::errors::Result;
use crate::graphs::{Context, Graph};

use serde::{Deserialize, Serialize};

use super::approx_pointwise::{create_approximation, PWLConfig};

/// A structure that defines the custom operation ApproxGeluDerivative that computes an approximate Gelu'(x / (2 ** precision)) * (2 ** precision) using piecewise-linear approximation.
///
/// So far this operation supports only INT64 scalar type.
/// For background on gelu function, see <https://arxiv.org/pdf/1606.08415v4.pdf>.
///
/// # Custom operation arguments
///
/// - Node containing a signed 64-bit array or scalar to compute the GELU'
///
/// # Custom operation returns
///
/// New ApproxGeluDerivative node
///
/// # Example
///
/// ```
/// # use ciphercore_base::graphs::create_context;
/// # use ciphercore_base::data_types::{scalar_type, array_type, INT64};
/// # use ciphercore_base::custom_ops::{CustomOperation};
/// # use ciphercore_base::ops::pwl::approx_gelu_derivative::ApproxGeluDerivative;
/// let c = create_context().unwrap();
/// let g = c.create_graph().unwrap();
/// let t = array_type(vec![3], INT64);
/// let x = g.input(t.clone()).unwrap();
/// let n = g.custom_op(CustomOperation::new(ApproxGeluDerivative {precision: 4}), vec![x]).unwrap();
///
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct ApproxGeluDerivative {
    /// Assume that we're operating in fixed precision arithmetic with denominator 2 ** precision.
    pub precision: u64,
}

#[typetag::serde]
impl CustomOperationBody for ApproxGeluDerivative {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        if arguments_types.len() != 1 {
            return Err(runtime_error!(
                "Invalid number of arguments for ApproxGeluDerivative"
            ));
        }
        let t = arguments_types[0].clone();
        if !t.is_scalar() && !t.is_array() {
            return Err(runtime_error!(
                "Argument in ApproxGeluDerivative must be a scalar or an array"
            ));
        }
        let sc = t.get_scalar_type();
        if sc != INT64 {
            return Err(runtime_error!(
                "Argument in ApproxGeluDerivative must consist of INT64's"
            ));
        }
        if self.precision > 30 || self.precision == 0 {
            return Err(runtime_error!("`precision` should be in range [1, 30]."));
        }

        let g = context.create_graph()?;
        let arg = g.input(t)?;
        // One can estimate max absolute error for a specific `log_buckets` by running `test_approx_gelu_derivative_array`
        // with sufficiently big `STEPS` constant.
        //
        // For `precision=15` we got such results.
        // In the case of `log_buckets=4` the max absolute error is around 0.024
        // In the case of `log_buckets=5` the max absolute error is around 0.006
        // In the case of `log_buckets=6` the max absolute error is around 0.0015
        let result = create_approximation(
            arg,
            approximate_gelu_derivative,
            -4.0,
            4.0,
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
        format!("ApproxGeluDerivative(scaling_factor=2**{})", self.precision)
    }
}

// We use approximation from https://arxiv.org/pdf/2104.02523.pdf
fn approximate_gelu_derivative(x: f32) -> f32 {
    let x3 = x.powi(3);
    0.5 * (0.0356774 * x3 + 0.797885 * x).tanh()
        + 0.5
        + (0.0535161 * x3 + 0.398942 * x) * (0.0356774 * x3 + 0.797885 * x).cosh().powi(-2)
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
        let context = simple_context(|g| {
            let i = g.input(scalar_type(INT64))?;
            g.custom_op(
                CustomOperation::new(ApproxGeluDerivative { precision }),
                vec![i],
            )
        })?;
        let mapped_c = run_instantiation_pass(context)?;
        let result = random_evaluate(
            mapped_c.get_context().get_main_graph()?,
            vec![Value::from_scalar(arg, INT64)?],
        )?;
        let res = result.to_i64(INT64)?;
        Ok(res)
    }

    const PRECISION_BITS: u64 = 15;
    const SCALE: f32 = (1 << PRECISION_BITS) as f32;
    const MAX_COORD_CHECK: i64 = 5 * (1 << PRECISION_BITS);
    const EXPECTED_MAX_ABS_ERROR: f32 = 0.006;

    fn array_helper(arg: Vec<i64>) -> Result<Vec<i64>> {
        let array_t = array_type(vec![arg.len() as u64], INT64);
        let context = simple_context(|g| {
            let i = g.input(array_t.clone())?;
            g.custom_op(
                CustomOperation::new(ApproxGeluDerivative {
                    precision: PRECISION_BITS,
                }),
                vec![i],
            )
        })?;
        let mapped_c = run_instantiation_pass(context)?;
        let result = random_evaluate(
            mapped_c.get_context().get_main_graph()?,
            vec![Value::from_flattened_array(&arg, INT64)?],
        )?;
        result.to_flattened_array_i64(array_t)
    }

    #[test]
    fn test_approx_gelu_derivative_scalar() {
        for i in (-MAX_COORD_CHECK..MAX_COORD_CHECK).step_by(1000) {
            let expected = (approximate_gelu_derivative((i as f32) / SCALE) * SCALE) as i64;
            let actual = scalar_helper(i, PRECISION_BITS).unwrap();
            let absolute_error = ((expected - actual).abs() as f32) / SCALE;
            assert!(absolute_error <= EXPECTED_MAX_ABS_ERROR);
        }
    }

    #[test]
    fn test_approx_gelu_derivative_array() {
        const STEPS: usize = 314;
        let arr: Vec<i64> = (-MAX_COORD_CHECK..MAX_COORD_CHECK)
            .step_by(MAX_COORD_CHECK as usize / STEPS)
            .collect();
        let res = array_helper(arr.clone()).unwrap();
        let mut max_abs_error = 0.0;
        for i in 0..arr.len() {
            let expected = (approximate_gelu_derivative((arr[i] as f32) / SCALE) * SCALE) as i64;
            let actual = res[i];
            let absolute_error = ((expected - actual).abs() as f32) / SCALE;
            if absolute_error > max_abs_error {
                max_abs_error = absolute_error;
            }
        }
        assert!(max_abs_error <= EXPECTED_MAX_ABS_ERROR);
    }
}
