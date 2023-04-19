//! AUC score metric, functionally equivalent to sklearn.metrics.roc_auc_score.
use std::cmp::Ordering;

use crate::custom_ops::{CustomOperation, CustomOperationBody};
use crate::data_types::{Type, INT128, INT64};
use crate::errors::Result;
use crate::graphs::{Context, Graph, Node, SliceElement};

use serde::{Deserialize, Serialize};

use super::fixed_precision::fixed_precision_config::FixedPrecisionConfig;
use super::goldschmidt_division::GoldschmidtDivision;
use super::integer_key_sort::SortByIntegerKey;
use super::utils::constant_scalar;

const MAX_LOG_ARRAY_SIZE: u64 = 20;

/// A structure that defines the custom operation AucMetric Area Under the Receiver Operating Characteristic Curve (ROC AUC).
///
/// This is the most commonly used metric for binary classification problems.
/// See also <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html>.
///
///
/// # Custom operation arguments
///
/// - Node containing a signed 1-dimensional 64-bit array with labels (0 or 1, possibly in the fixed precision format);
/// - Node containing a 1-dimensional 64-bit array with predictions (can be integers or fixed precision, only relative order matters).
///
/// # Custom operation returns
///
/// New AucScore node
///
/// # Example
///
/// ```
/// # use ciphercore_base::graphs::create_context;
/// # use ciphercore_base::data_types::{scalar_type, array_type, INT64};
/// # use ciphercore_base::custom_ops::CustomOperation;
/// # use ciphercore_base::ops::auc::AucScore;
/// # use ciphercore_base::ops::fixed_precision::fixed_precision_config::FixedPrecisionConfig;
/// let c = create_context().unwrap();
/// let g = c.create_graph().unwrap();
/// let t = array_type(vec![5], INT64);
/// let y_true = g.input(t.clone()).unwrap();
/// let y_pred = g.input(t.clone()).unwrap();
/// let fp = FixedPrecisionConfig::default();
/// let auc = g.custom_op(CustomOperation::new(AucScore {fp}), vec![y_true, y_pred]).unwrap();
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct AucScore {
    /// Fixed precision config for labels.
    pub fp: FixedPrecisionConfig,
}

#[typetag::serde]
impl CustomOperationBody for AucScore {
    fn instantiate(&self, context: Context, arguments_types: Vec<Type>) -> Result<Graph> {
        if arguments_types.len() != 2 {
            return Err(runtime_error!("Invalid number of arguments for AucMetric"));
        }
        let t = arguments_types[0].clone();
        if !t.is_array() {
            return Err(runtime_error!(
                "`y_true` in AucMetric must be an array, got {t:?}"
            ));
        }
        if t.get_dimensions().len() != 1 {
            return Err(runtime_error!(
                "`y_true` in AucMetric must be 1-dimensional, got {t:?}"
            ));
        }
        let n = t.get_dimensions()[0];
        if n >= (1 << MAX_LOG_ARRAY_SIZE) {
            return Err(runtime_error!(
                "`y_true` in AucMetric must have less than 2^{} elements, got {n:?}",
                MAX_LOG_ARRAY_SIZE
            ));
        }
        let sc = t.get_scalar_type();
        if sc != INT64 {
            return Err(runtime_error!(
                "`y_true` in AucMetric must consist of INT64's, got {sc:?}"
            ));
        }
        if arguments_types[1] != t {
            return Err(runtime_error!(
                "`y_pred` in AucMetric must be of the same type as `y_true`, got {:?} vs {:?}",
                t,
                arguments_types[1]
            ));
        }

        let g = context.create_graph()?;
        let y_true = g.input(t.clone())?;
        let y_pred = g.input(t)?;
        // Intuitively, AUC is the probability that a randomly chosen positive example is ranked higher than a randomly chosen negative example.
        // SKLearn definition expands it with the following: if two examples have the same prediction, this pair is accounted as 0.5 rather than 1.
        // E.g., if we have y_true = [0, 0, 1, 1], and y_pred = [10, 10, 10, 10], SKLearn will return 0.5.
        // This complicates implementation in CipherCore. Luckily, we can utilize the following trick.
        // Let's assume that we compute the AUC by sorting the data over predictions, and ignoring the fact that some of the predictions are equal.
        // Our sorting is stable, so if we repeat the process on the reversed data, pairs of equal predictions will be counted as 1 in one case, and
        // 0 in the other one, while pairs of predictions which are not equal will be counted the same way in both cases.
        // This means that we can compute AUC in a naive way on the data and reversed data, and just average the results.
        let auc1 = compute_naive_auc(y_true.clone(), y_pred.clone(), &self.fp)?;
        let y_true = y_true.get_slice(vec![SliceElement::SubArray(None, None, Some(-1))])?;
        let y_pred = y_pred.get_slice(vec![SliceElement::SubArray(None, None, Some(-1))])?;
        let auc2 = compute_naive_auc(y_true, y_pred, &self.fp)?;
        let auc = auc1.add(auc2)?.truncate(2)?;
        auc.set_as_output()?;
        g.finalize()?;
        Ok(g)
    }

    fn get_name(&self) -> String {
        format!("AucScore(fp={:?})", self.fp)
    }
}

fn compute_naive_auc(y_true: Node, y_pred: Node, fp: &FixedPrecisionConfig) -> Result<Node> {
    // This function doesn't account for equal predictions.
    // Assuming all predictions are distinct:
    // AUC = (number of pairs (i, j) such that y_pred[i] < y_pred[j] and y_true[i] < y_true[j]) / (number of pairs (i, j) such that y_true[i] < y_true[j]).
    let g = y_true.get_graph();
    let joined =
        g.create_named_tuple(vec![("y_pred".into(), y_pred), ("y_true".into(), y_true)])?;
    let joined = g.custom_op(
        CustomOperation::new(SortByIntegerKey {
            key: "y_pred".into(),
        }),
        vec![joined],
    )?;
    let y_true = joined.named_tuple_get("y_true".into())?;

    // Compute AUC denominator, i.e. num_ones * num_zeros.
    let num_ones = y_true.sum(vec![0])?.truncate(fp.denominator())?;
    let n = y_true.get_type()?.get_dimensions()[0] as i64;
    let n = constant_scalar(&g, n, INT64)?;
    let num_zeros = n.subtract(num_ones.clone())?;
    let denominator = num_ones.multiply(num_zeros)?;

    // Compute AUC numerator.
    let one = constant_scalar(&g, fp.denominator(), INT64)?;
    let num_zeros_on_prefix = one.subtract(y_true.clone())?.cum_sum(0)?;
    let num_zeros_before_one = num_zeros_on_prefix
        .multiply(y_true)?
        .truncate(fp.denominator())?;
    let numerator = num_zeros_before_one
        .sum(vec![0])?
        .truncate(fp.denominator())?;

    // Compute AUC.
    let numerator = i64_to_i128(numerator)?;
    let denominator = i64_to_i128(denominator)?;
    let denom_bits = MAX_LOG_ARRAY_SIZE * 2;
    // Note: GoldschmidtDivision requires denominator to be smaller than 2 ** denominator_cap_2k.
    // We use 2 ** 40 as a hard-coded constant, meaning that sqrt(denominator) must be smaller than 2 ** 20 (imposing restriction on the number of elements in the array).
    // We can potentially support larger arrays, if we first divide by num_ones, and then - by num_zeros.
    let result = g.custom_op(
        CustomOperation::new(GoldschmidtDivision {
            // Use 7 = log_2(128) iterations.
            iterations: 7,
            denominator_cap_2k: denom_bits,
        }),
        vec![numerator, denominator],
    )?;

    let result = match denom_bits.cmp(&fp.fractional_bits) {
        Ordering::Less => result.multiply(constant_scalar(
            &g,
            1 << (fp.fractional_bits - denom_bits),
            INT128,
        )?)?,
        Ordering::Equal => result,
        Ordering::Greater => result.truncate(1 << (denom_bits - fp.fractional_bits))?,
    };

    i128_to_i64(result)
}

fn i64_to_i128(x: Node) -> Result<Node> {
    let g = x.get_graph();
    let bits = x.a2b()?;
    let zeros = g.zeros(bits.get_type()?)?;
    let bits = g.concatenate(vec![bits, zeros], 0)?;
    bits.b2a(INT128)
}

fn i128_to_i64(x: Node) -> Result<Node> {
    let bits = x.a2b()?;
    let bits = bits.get_slice(vec![SliceElement::SubArray(None, Some(64), None)])?;
    bits.b2a(INT64)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::custom_ops::run_instantiation_pass;
    use crate::custom_ops::CustomOperation;
    use crate::data_types::array_type;
    use crate::data_values::Value;
    use crate::evaluators::random_evaluate;
    use crate::graphs::util::simple_context;

    fn test_helper(y_true: Vec<i64>, y_pred: Vec<i64>) -> Result<f64> {
        let array_t = array_type(vec![y_true.len() as u64], INT64);
        let c = simple_context(|g| {
            let y_true = g.input(array_t.clone())?;
            let y_pred = g.input(array_t)?;
            g.custom_op(
                CustomOperation::new(AucScore {
                    fp: FixedPrecisionConfig::default(),
                }),
                vec![y_true, y_pred],
            )
        })?;
        let mapped_c = run_instantiation_pass(c)?;
        let result = random_evaluate(
            mapped_c.get_context().get_main_graph()?,
            vec![
                Value::from_flattened_array(&y_true, INT64)?,
                Value::from_flattened_array(&y_pred, INT64)?,
            ],
        )?;
        Ok(result.to_i64(INT64)? as f64 / FixedPrecisionConfig::default().denominator_f64())
    }

    #[test]
    fn test_auc_simple_case() -> Result<()> {
        let one = FixedPrecisionConfig::default().denominator() as i64;
        let y_true = vec![0, one, 0, one];
        let y_pred = vec![-10, 30, 20, 10];
        let res = test_helper(y_true, y_pred)?;
        assert!((res - 0.75).abs() < 1e-3);
        Ok(())
    }

    #[test]
    fn test_auc_equal_predictions() -> Result<()> {
        let one = FixedPrecisionConfig::default().denominator() as i64;
        let y_true = vec![0, one, 0, one];
        let y_pred = vec![42, 42, 42, 42];
        let res = test_helper(y_true, y_pred)?;
        assert!((res - 0.5).abs() < 1e-3);
        Ok(())
    }

    #[test]
    fn test_auc_large_array() -> Result<()> {
        let one = FixedPrecisionConfig::default().denominator() as i64;
        let mut y_true = vec![];
        let mut y_pred = vec![];
        for i in 0..10000 {
            y_true.push(if i < 5000 { 0 } else { one });
            y_pred.push(i);
        }
        let res = test_helper(y_true, y_pred)?;
        assert!((res - 1.0).abs() < 1e-3);
        Ok(())
    }
}
