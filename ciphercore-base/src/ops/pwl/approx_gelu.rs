//! GELU(x) piecewise-linear approximation.
use crate::custom_ops::CustomOperationBody;
use crate::data_types::{Type, INT64};
use crate::errors::Result;
use crate::graphs::{Context, Graph};

use serde::{Deserialize, Serialize};

use super::approx_pointwise::{create_approximation, PWLConfig};

/// A structure that defines the custom operation ApproxGelu that computes an approximate Gelu(x / (2 ** precision)) * (2 ** precision) using piecewise-linear approximation.
///
/// So far this operation supports only INT64 scalar type.
/// For background on gelu function, see <https://arxiv.org/pdf/1606.08415v4.pdf>.
///
/// # Custom operation arguments
///
/// - Node containing a signed 64-bit array or scalar to compute the GELU
///
/// # Custom operation returns
///
/// New ApproxGelu node
///
/// # Example
///
/// ```
/// # use ciphercore_base::graphs::create_context;
/// # use ciphercore_base::data_types::{scalar_type, array_type, INT64};
/// # use ciphercore_base::custom_ops::{CustomOperation};
/// # use ciphercore_base::ops::pwl::approx_gelu::ApproxGelu;
/// let c = create_context().unwrap();
/// let g = c.create_graph().unwrap();
/// let t = array_type(vec![3], INT64);
/// let x = g.input(t.clone()).unwrap();
/// let n = g.custom_op(CustomOperation::new(ApproxGelu {precision: 4}), vec![x]).unwrap();
///
// TODO: generalize to other types.
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct ApproxGelu {
    /// Assume that we're operating in fixed precision arithmetic with denominator 2 ** precision.
    pub precision: u64,
}

#[typetag::serde]
impl CustomOperationBody for ApproxGelu {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        if arguments_types.len() != 1 {
            return Err(runtime_error!("Invalid number of arguments for ApproxGelu"));
        }
        let t = arguments_types[0].clone();
        if !t.is_scalar() && !t.is_array() {
            return Err(runtime_error!(
                "Argument in ApproxGelu must be a scalar or an array"
            ));
        }
        let sc = t.get_scalar_type();
        if sc != INT64 {
            return Err(runtime_error!(
                "Argument in ApproxGelu must consist of INT64's"
            ));
        }
        if self.precision > 30 || self.precision == 0 {
            return Err(runtime_error!("`precision` should be in range [1, 30]."));
        }

        let g = context.create_graph()?;
        let arg = g.input(t)?;
        // Choice of parameters:
        // -- left/right: our typical use-case is precision=15, leading to minimum value around 3e-5. GELU(-5) is already lower than that, so -4 is a reasonable choice with our precision. On the other side, at 4, Gelu is pretty much linear;
        // -- log_buckets: we look at max absolute difference to the real sigmoid. It looks as follows:
        //    log_buckets=4 => 0.0232,
        //    log_buckets=5 => 0.0059,
        //    log_buckets=6 => 0.0015.
        // After 5 segments, we're getting diminishing returns, so it doesn't make sense to go higher (for the sake of performance).
        // -- flatten_left/flatten_right: GELU is linear on the right and flat on the left.
        let result = create_approximation(
            arg,
            approximate_gelu,
            -4.0,
            4.0,
            self.precision,
            PWLConfig {
                log_buckets: 5,
                flatten_left: true,
                flatten_right: false,
            },
        )?;
        result.set_as_output()?;
        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        format!("ApproxGelu(scaling_factor=2**{})", self.precision)
    }
}

fn approximate_gelu(x: f32) -> f32 {
    // It appears there is no Erf in Rust without additional crates. So we use an approximation.
    // See also <https://paperswithcode.com/method/gelu>.
    // The accurate GELU formula is: 0.5 * x * (1 + erf(x / sqrt(2))).
    let tanh_arg = (2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
    let ex = tanh_arg.exp();
    let emx = (-tanh_arg).exp();
    let tanh = (ex - emx) / (ex + emx);
    0.5 * x * (1.0 + tanh)
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
            g.custom_op(CustomOperation::new(ApproxGelu { precision }), vec![i])
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
            g.custom_op(CustomOperation::new(ApproxGelu { precision: 10 }), vec![i])
        })?;
        let mapped_c = run_instantiation_pass(c)?;
        let result = random_evaluate(
            mapped_c.get_context().get_main_graph()?,
            vec![Value::from_flattened_array(&arg, INT64)?],
        )?;
        result.to_flattened_array_i64(array_t)
    }

    #[test]
    fn test_approx_gelu_scalar() {
        for i in (-5000..5000).step_by(1000) {
            let expected = (approximate_gelu((i as f32) / 1024.0) * 1024.0) as i64;
            let actual = scalar_helper(i, 10).unwrap();
            let absolute_error = ((expected - actual).abs() as f64) / 1024.0;
            assert!(absolute_error <= 0.01);
        }
    }

    #[test]
    fn test_approx_gelu_array() {
        let arr: Vec<i64> = (-5000..5000).step_by(100).collect();
        let res = array_helper(arr.clone()).unwrap();
        for i in 0..arr.len() {
            let expected = (approximate_gelu((arr[i] as f32) / 1024.0) * 1024.0) as i64;
            let actual = res[i];
            let absolute_error = ((expected - actual).abs() as f64) / 1024.0;
            assert!(absolute_error <= 0.01);
        }
    }
}
