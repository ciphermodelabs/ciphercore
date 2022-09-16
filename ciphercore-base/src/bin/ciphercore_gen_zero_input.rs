//! Code of a binary that generates JSON of zero inputs for a given context

use ciphercore_utils::execute_main::execute_main;

use ciphercore_base::data_values::Value;
use ciphercore_base::errors::Result;
use ciphercore_base::graphs::{Context, Operation};
use ciphercore_base::typed_value::TypedValue;

use clap::Parser;
use std::fs;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about=None)]
struct Args {
    #[clap(value_parser)]
    /// Path to file that contains the input context
    input_context: String,
}

/// This binary reads the context from a given file and prints JSON for zero inputs of appropriate types.
///
/// # Arguments
///
/// * `inputs_path` - path to file that contains the context
///
/// # Usage
///
/// < this_binary > <inputs_path>
fn main() {
    env_logger::init();
    execute_main(|| -> Result<()> {
        let args = Args::parse();
        let context = serde_json::from_str::<Context>(&fs::read_to_string(args.input_context)?)?;
        let mut inputs = vec![];
        for node in context.get_main_graph()?.get_nodes() {
            if let Operation::Input(ref input_type) = node.get_operation() {
                inputs.push(TypedValue::new(
                    input_type.clone(),
                    Value::zero_of_type(input_type.clone()),
                )?);
            }
        }
        println!("{}", serde_json::to_string(&inputs)?);
        Ok(())
    });
}
