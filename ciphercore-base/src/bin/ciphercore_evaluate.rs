#![cfg_attr(feature = "nightly-features", feature(backtrace))]

use ciphercore_base::errors::Result;
use ciphercore_base::evaluators::get_result_util::{get_evaluator_result, parse_json_array};
use ciphercore_base::evaluators::simple_evaluator::SimpleEvaluator;
use ciphercore_base::graphs::Context;
use ciphercore_utils::execute_main::execute_main;
use std::fs;

use clap::Parser;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about=None)]
struct Args {
    context_path: String,
    inputs_path: String,
    #[clap(long)]
    reveal_output: bool,
}

fn main() {
    env_logger::init();
    execute_main(|| -> Result<()> {
        let args = Args::parse();
        let serialized_context = fs::read_to_string(&args.context_path)?;
        let raw_context = serde_json::from_str::<Context>(&serialized_context)?;
        let json_inputs = fs::read_to_string(&args.inputs_path)?;
        let json_inputs = json::parse(&json_inputs)?;
        let inputs = parse_json_array(&json_inputs)?;
        let result = get_evaluator_result(
            raw_context,
            inputs,
            args.reveal_output,
            SimpleEvaluator::new(None)?,
        )?;
        if args.reveal_output {
            eprintln!("Revealing the output");
        }
        println!("{}", result.to_json()?);
        Ok(())
    });
}
