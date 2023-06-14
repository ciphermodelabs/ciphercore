//! Code of a binary printing evaluated result of a given serialized context on given context inputs

use ciphercore_base::errors::Result;
use ciphercore_base::evaluators::get_result_util::get_evaluator_result;
use ciphercore_base::evaluators::simple_evaluator::SimpleEvaluator;
use ciphercore_base::graphs::Context;
use ciphercore_base::typed_value::TypedValue;
use ciphercore_utils::eprintln_or_log;
use ciphercore_utils::execute_main::execute_main;
use std::fs;

use clap::Parser;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about=None)]
struct Args {
    #[clap(value_parser)]
    /// Path to a file containing serialized context
    context_path: String,
    #[clap(value_parser)]
    /// Path to a file containing serialized inputs
    inputs_path: String,
    #[clap(long, value_parser)]
    /// (optional) Boolean to indicate if the output is to be revealed for secret-shared outputs
    reveal_output: bool,
}

/// This binary evaluates a given context over the provided inputs using a simple evaluator.
///
/// For a secret-shared output, output is revealed if the `reveal_output` boolean binary argument is set to `true`.
///
/// # Arguments
///
/// * `context_path` - path to a serialized context
/// * `inputs_path` - path to serialized input(s)
/// * `reveal_output` - boolean to indicate if output is to be revealed
///
/// # Usage
///
/// < this_binary > [--reveal-output] <CONTEXT_PATH> <INPUTS_PATH>
fn main() {
    // Initialize a logger that collects information about errors and panics within CipherCore.
    // This information can be accessed via RUST_LOG.
    env_logger::init();
    // Execute CipherCore code such that all the internal errors are properly formatted and logged.
    execute_main(|| -> Result<()> {
        // Parse the input arguments
        let args = Args::parse();
        // Read the entire file containing a serialized context as a string
        let serialized_context = fs::read_to_string(&args.context_path)?;
        // Deserialize into a context object from the serialized context string
        let raw_context = serde_json::from_str::<Context>(&serialized_context)?;
        // Read the entire file containing serialized inputs as a string
        let json_inputs = fs::read_to_string(&args.inputs_path)?;
        // Parse inputs to obtain a vector of typed values, i.e., pair of type, and its value
        let inputs = serde_json::from_str::<Vec<TypedValue>>(&json_inputs)?;
        // Use the simple evaluator to obtain the typed result value
        let result = get_evaluator_result(
            raw_context,
            inputs,
            args.reveal_output,
            SimpleEvaluator::new(None)?,
        )?;
        // Depending on the input argument, print whether the output is revealed
        if args.reveal_output {
            eprintln_or_log!("Revealing the output");
        }
        // Print the serialized result to stdout
        println!("{}", serde_json::to_string(&result)?);
        Ok(())
    });
}
