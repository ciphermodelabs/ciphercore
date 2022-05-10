#![cfg_attr(feature = "nightly-features", feature(backtrace))]
#[macro_use]
extern crate ciphercore_base;

use std::fs;

use ciphercore_base::errors::Result;
use ciphercore_base::evaluators::simple_evaluator::SimpleEvaluator;
use ciphercore_base::graphs::Context;
use ciphercore_base::inline::inline_common::DepthOptimizationLevel;
use ciphercore_base::inline::inline_ops::{InlineConfig, InlineMode};
use ciphercore_base::mpc::mpc_compiler::{compile_context, IOStatus};
use ciphercore_utils::execute_main::execute_main;

use clap::{ArgEnum, Parser};
use std::collections::HashSet;

#[derive(Debug, ArgEnum, Clone)]
enum InlineModeArg {
    Simple,
    DepthOptimizedDefault,
    DepthOptimizedExtreme,
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about=None)]
struct Args {
    /// Path to file that contains the serialized Context
    context_path: String,
    #[clap(arg_enum)]
    inline_mode: InlineModeArg,
    input_parties: String,
    output_parties: String,
}

fn get_tokens(s: String) -> Result<Vec<IOStatus>> {
    let tokens: Vec<String> = s.split(',').map(|x| x.to_owned()).collect();
    if tokens.is_empty() {
        return Err(runtime_error!("Empty tokens"));
    }
    let mut result = Vec::new();
    for token in tokens {
        let tmp: &str = &token;
        match tmp {
            "0" => {
                result.push(IOStatus::Party(0));
            }
            "1" => {
                result.push(IOStatus::Party(1));
            }
            "2" => {
                result.push(IOStatus::Party(2));
            }
            "public" => {
                result.push(IOStatus::Public);
            }
            "secret-shared" => {
                result.push(IOStatus::Shared);
            }
            _ => {
                return Err(runtime_error!("Invalid token: {}", token));
            }
        }
    }
    Ok(result)
}

fn parse_input_parties(s: String) -> Result<Vec<IOStatus>> {
    let tokens = get_tokens(s)?;
    Ok(tokens)
}

fn parse_output_parties(s: String) -> Result<Vec<IOStatus>> {
    let tokens = get_tokens(s)?;
    if tokens.len() == 1 {
        match tokens[0] {
            IOStatus::Party(_) => Ok(tokens),
            IOStatus::Public => Ok(vec![
                IOStatus::Party(0),
                IOStatus::Party(1),
                IOStatus::Party(2),
            ]),
            IOStatus::Shared => Ok(vec![]),
        }
    } else {
        let mut parties = HashSet::new();
        for token in &tokens {
            match token {
                IOStatus::Party(party_id) => {
                    if parties.contains(&party_id) {
                        return Err(runtime_error!("Invalid output parties: duplicate party"));
                    }
                    parties.insert(party_id);
                }
                _ => {
                    return Err(runtime_error!("Invalid output parties"));
                }
            }
        }
        Ok(tokens)
    }
}

fn get_inline_mode(mode: InlineModeArg) -> InlineMode {
    match mode {
        InlineModeArg::Simple => InlineMode::Simple,
        InlineModeArg::DepthOptimizedDefault => {
            InlineMode::DepthOptimized(DepthOptimizationLevel::Default)
        }
        InlineModeArg::DepthOptimizedExtreme => {
            InlineMode::DepthOptimized(DepthOptimizationLevel::Extreme)
        }
    }
}

fn get_evaluator() -> Result<SimpleEvaluator> {
    SimpleEvaluator::new(None)
}

fn main() {
    env_logger::init();
    execute_main(|| -> Result<()> {
        let args = Args::parse();
        let serialized_context = fs::read_to_string(&args.context_path)?;
        let context = serde_json::from_str::<Context>(&serialized_context)?;
        let input_parties = parse_input_parties(args.input_parties)?;
        let output_parties = parse_output_parties(args.output_parties)?;
        let compiled_context = compile_context(
            context,
            input_parties,
            output_parties,
            InlineConfig {
                default_mode: get_inline_mode(args.inline_mode),
                ..Default::default()
            },
            get_evaluator,
        )?;
        println!("{}", serde_json::to_string(&compiled_context).unwrap());
        Ok(())
    });
}
