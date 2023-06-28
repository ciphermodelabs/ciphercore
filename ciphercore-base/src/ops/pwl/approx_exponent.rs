//! Exp(x) piecewise-linear approximation.
use crate::custom_ops::CustomOperationBody;
use crate::data_types::{Type, INT64};
use crate::errors::Result;
use crate::graphs::{Context, Graph};

use serde::{Deserialize, Serialize};

use super::approx_pointwise::{create_approximation, PWLConfig};

/// A structure that defines the custom operation ApproxExponent that computes an approximate exp(x / (2 ** precision)) * (2 ** precision) using piecewise-linear approximation.
///
/// So far this operation supports only INT64 scalar type.
///
/// # Custom operation arguments
///
/// - Node containing a signed 64-bit array or scalar to compute the exponent
///
/// # Custom operation returns
///
/// New ApproxExponent node
///
/// # Example
///
/// ```
/// # use ciphercore_base::graphs::create_context;
/// # use ciphercore_base::data_types::{scalar_type, array_type, INT64};
/// # use ciphercore_base::custom_ops::{CustomOperation};
/// # use ciphercore_base::ops::pwl::approx_exponent::ApproxExponent;
/// let c = create_context().unwrap();
/// let g = c.create_graph().unwrap();
/// let t = array_type(vec![3], INT64);
/// let x = g.input(t.clone()).unwrap();
/// let n = g.custom_op(CustomOperation::new(ApproxExponent {precision: 4}), vec![x]).unwrap();
///
// TODO: generalize to other types.
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct ApproxExponent {
    /// Assume that we're operating in fixed precision arithmetic with denominator 2 ** precision.
    pub precision: u64,
}

#[typetag::serde]
impl CustomOperationBody for ApproxExponent {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        if arguments_types.len() != 1 {
            return Err(runtime_error!(
                "Invalid number of arguments for ApproxExponent"
            ));
        }
        let t = arguments_types[0].clone();
        if !t.is_scalar() && !t.is_array() {
            return Err(runtime_error!(
                "Argument in ApproxExponent must be a scalar or an array"
            ));
        }
        let sc = t.get_scalar_type();
        if sc != INT64 {
            return Err(runtime_error!(
                "Argument in ApproxExponent must consist of INT64's"
            ));
        }
        if self.precision > 30 || self.precision == 0 {
            return Err(runtime_error!("`precision` should be in range [1, 30]."));
        }

        let g = context.create_graph()?;
        let arg = g.input(t)?;
        // Choice of parameters:
        // -- left/right: our typical use-case is precision=15, leading to minimum value around 3e-5. Exp(-10) is 4e-5, so right=-left=10 is a reasonable choice with our precision;
        // -- log_buckets: we look at max relative difference to the real exponent. It looks as follows (note that this usually happens around -10, for higher values, it is more accurate):
        //    log_buckets=4 => 36%,
        //    log_buckets=5 => 21%,
        //    log_buckets=6 => 22%,
        //    log_buckets=7 => 22%.
        // Note that it is not monotonic due to numerical issues for very low values. From this table, the best value is 5.
        // -- flatten_left/flatten_right: exponent is flat on the left, so we replicate this in our approximation (we don't want to go to 0 and below).
        // However, due to the issues with Truncate on non-power-of-2 denominators, we want (right - left) to be a power of 2,
        // so we round the boundaries to +/- 16, and increase the log_buckets to 6.
        let result = create_approximation(
            arg,
            |x| x.exp(),
            // The boundaries are chosen in a way that (left - right) * precision is a power of 2.
            // It is important because otherwise we get non-power-of-2 truncations during the approximation.
            -16.0,
            16.0,
            self.precision,
            PWLConfig {
                log_buckets: 6,
                flatten_left: true,
                flatten_right: false,
            },
        )?;
        result.set_as_output()?;
        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        format!("ApproxExponent(scaling_factor=2**{})", self.precision)
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
            g.custom_op(CustomOperation::new(ApproxExponent { precision }), vec![i])
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
                CustomOperation::new(ApproxExponent { precision: 10 }),
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

    #[test]
    fn test_approx_exp_scalar() {
        for i in [-10000, -1000, -100, -1, 0, 1, 100, 1000, 10000] {
            let expected = (((i as f64) / 1024.0).exp() * 1024.0) as i64;
            let actual = scalar_helper(i, 10).unwrap();
            let relative_error = ((expected - actual).abs() as f64)
                / (1.0 + f64::max(expected as f64, actual as f64));
            assert!(relative_error <= 0.05);
        }
    }

    #[test]
    fn test_approx_exp_array() {
        let arr = vec![23, 32, 57, 1271, 183, 555, -23, -32, -57, -1271, -183, -555];
        let res = array_helper(arr.clone()).unwrap();
        for i in 0..arr.len() {
            let expected = (((arr[i] as f64) / 1024.0).exp() * 1024.0) as i64;
            let actual = res[i];
            let relative_error = ((expected - actual).abs() as f64)
                / (1.0 + f64::max(expected as f64, actual as f64));
            assert!(relative_error <= 0.05);
        }
    }
}
