use crate::custom_ops::run_instantiation_pass;
use crate::data_types::Type;
use crate::evaluators::Evaluator;
use crate::graphs::{Context, Operation};
use crate::random::PRNG;

use crate::errors::Result;
use crate::typed_value::TypedValue;
use json::JsonValue;

#[doc(hidden)]
/// Parses the JSON input to return a vector of type value
///
/// Runtime error is returned if input JSON value is not of array type.
///
/// This function is for internal use.
///
/// # Arguments
///
/// `j` - reference to a JSON value
///
/// # Returns
///
/// Vector of typed values
///
/// ## Example
///
/// ```
/// # use json::{JsonValue, object};
/// # use ciphercore_base::evaluators::get_result_util::parse_json_array;
/// # use ciphercore_base::typed_value::TypedValue;
/// # use ciphercore_base::data_types::{INT64, scalar_type};
/// # use ciphercore_base::data_values::Value;
///
/// let mut array = JsonValue::new_array();
/// let val1: i64 = -123123;
/// array.push(object!{
///     kind: "scalar",
///     type: "i64",
///     value: val1,
/// });
/// let tv_sc_i64 = TypedValue::new(
///     scalar_type(INT64),
///     Value::from_scalar(val1, INT64).unwrap()).unwrap();
///
/// let typed_val = parse_json_array(&array).unwrap();
/// assert_eq!(typed_val[0], tv_sc_i64);
/// ```
pub fn parse_json_array(j: &JsonValue) -> Result<Vec<TypedValue>> {
    if let JsonValue::Array(a) = j {
        let mut result = vec![];
        for element in a {
            result.push(TypedValue::from_json(element)?);
        }
        Ok(result)
    } else {
        Err(runtime_error!("JSON array expected"))
    }
}

#[doc(hidden)]
/// This function evaluates a given context on given inputs using a given evaluator.
///
/// A flag may be passed if the output has to be revealed for secret-shared outputs.
///
/// Returns a runtime error if the number and type of inputs do not match with
/// those of the inputs of the context.
///
/// This function is for internal use.
///
/// # Arguments
///
/// * `context` - context to compute the result for
///
/// * `inputs` - vector containing type-value data
///
/// * `reveal_output` - boolean to indicate if output is to be revealed
///
/// * `evaluator` - evaluator to be used for processing the context
///
/// # Returns
///
/// Result corresponding to the inputs for the given context
///
/// ## Example
///
/// ```
/// # use ciphercore_base::graphs::{Context, create_context};
/// # use ciphercore_base::typed_value::TypedValue;
/// # use ciphercore_base::data_types::{UINT64, scalar_type};
/// # use ciphercore_base::data_values::Value;
/// # use ciphercore_base::evaluators::get_result_util::get_evaluator_result;
/// # use ciphercore_base::evaluators::simple_evaluator::SimpleEvaluator;
/// # use ciphercore_base::errors::Result;
/// let c = || -> Result<Context> {
///     let c = create_context()?;
///     let g = c.create_graph()?;
///     let t = scalar_type(UINT64);
///     let ip0 = g.input(t.clone())?;
///     let ip1 = g.input(t)?;
///     let r = g.add(ip0, ip1)?;
///     g.set_output_node(r)?;
///     g.finalize()?;
///     c.set_main_graph(g)?;
///     c.finalize();
///     Ok(c)
/// }().unwrap();
/// let inputs = || -> Result<Vec<TypedValue>> {
///     Ok(vec![
///     TypedValue::new(scalar_type(UINT64), Value::from_scalar(2, UINT64)?)?,
///     TypedValue::new(scalar_type(UINT64), Value::from_scalar(2, UINT64)?)?
///     ])
/// }().unwrap();
/// let result = get_evaluator_result(
///     c,
///     inputs,
///     false,
///     SimpleEvaluator::new(None).unwrap()
/// ).unwrap();
/// let tv_sc_u64 = TypedValue::new(
///     scalar_type(UINT64),
///     Value::from_scalar(4, UINT64).unwrap()).unwrap();
/// assert_eq!(result, tv_sc_u64);
/// ```
pub fn get_evaluator_result<T: Evaluator>(
    context: Context,
    inputs: Vec<TypedValue>,
    reveal_output: bool,
    mut evaluator: T,
) -> Result<TypedValue> {
    let context = run_instantiation_pass(context)?.get_context();
    let mut input_types = vec![];
    for node in context.get_main_graph()?.get_nodes() {
        if let Operation::Input(t) = node.get_operation() {
            input_types.push(t);
        }
    }
    if inputs.len() != input_types.len() {
        return Err(runtime_error!(
            "Invalid number of inputs: {} expected, {} received",
            input_types.len(),
            inputs.len()
        ));
    }
    let mut input_values = vec![];
    let mut prng = PRNG::new(None)?;
    for i in 0..inputs.len() {
        eprint!("Input {}: ", i);
        if inputs[i].value.check_type(input_types[i].clone())? {
            eprintln!("Using as is");
            input_values.push(inputs[i].value.clone());
        } else {
            let e = Err(runtime_error!("Invalid input value"));
            let v = if let Type::Tuple(v) = input_types[i].clone() {
                v
            } else {
                return e;
            };
            if v.len() == 3
                && v[0] == v[1]
                && v[0] == v[2]
                && inputs[i].value.check_type((*v[0]).clone())?
            {
                eprintln!("Secret-sharing");
                // This is a nasty hack.
                // It allows to pass an array of i32's to a graph
                // that accepts bit arrays etc.
                input_values.push(
                    TypedValue::new((*v[0]).clone(), inputs[i].value.clone())?
                        .secret_share(&mut prng)?
                        .value,
                );
            } else {
                return e;
            }
        }
    }

    evaluator.preprocess(context.clone())?;
    let mut result = TypedValue::new(
        context.get_main_graph()?.get_output_node()?.get_type()?,
        evaluator.evaluate_graph(context.get_main_graph()?, input_values)?,
    )?;
    if reveal_output {
        result = result.secret_share_reveal()?;
    }
    Ok(result)
}
