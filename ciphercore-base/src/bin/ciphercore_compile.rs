//! Code of a binary that outputs a serialized context, which is optimized and MPC-ready, from a given context
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
    /// Path to file that contains the serialized context
    context_path: String,
    #[clap(arg_enum)]
    /// Argument to specify the inline mode required.
    // Valid values are `simple`, `depth-optimized-default` and `depth-optimized-extreme`.
    inline_mode: InlineModeArg,
    /// String comprising comma separated list of input parties' IDs, valid ID values include `0`, `1`, `2` OR `public` OR `secret-shared`.
    input_parties: String,
    /// String comprising comma separated list of output parties' IDs, which could be `0`, `1`, and `2` OR `public` OR `secret-shared`.
    output_parties: String,
}

/// Returns tokens from given stringed input consisting of (1) comma separated party IDs, OR (2) "public", OR (3) "secret-shared".
///
/// # Arguments
///
/// `s` - string that encodes the information concerning participating parties
///
/// # Returns
///
/// Vector of party identifiers
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

/// Returns vector of tokens comprising party identifiers.
///
/// Wrapper function for obtaining identifier tokens.
///
/// # Arguments
///
/// 's' - string containing party identifiers
///
/// # Returns
///
/// Vector of party of identifiers
fn parse_input_parties(s: String) -> Result<Vec<IOStatus>> {
    let tokens = get_tokens(s)?;
    Ok(tokens)
}

/// Returns vector of party identifiers given a string.
///
/// # Arguments
///
/// `s` - string containing party identifiers
///
/// # Returns
///
/// Vector of party of identifiers
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

/// Returns internal inline mode based on the user-given argument
///
/// # Argument
///
/// `mode` - user-given inline mode argument
///
/// # Returns
///
/// Inline mode
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

/// Returns an unseeded simple evaluator instance
///
/// # Returns
///
/// Evaluator
fn get_evaluator() -> Result<SimpleEvaluator> {
    SimpleEvaluator::new(None)
}

/// This binary prints to stdout the serialized context, which is optimized and MPC-ready, from a given context.
///
/// # Arguments
///
/// * `context_path` - path to file that contains the serialized context
/// * `inline_mode` - argument to specify the inline mode required, valid values are `simple`, `depth-optimized-default` and  `depth-optimized-extreme`
/// * `input_parties` - string comprising of either a comma separated list of input parties' IDs, valid ID values include `0`, `1`, `2` OR `public` OR `secret-shared`
/// * `output_parties` - string comprising comma separated list of output parties' IDs, which could be `0`, `1`, and `2` OR `public` OR `secret-shared`
///
/// # Usage
///
/// < this_binary > <context_path> <inline_mode> <input_parties> <output_parties>
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
        let context = serde_json::from_str::<Context>(&serialized_context)?;
        // Parse the input party information
        let input_parties = parse_input_parties(args.input_parties)?;
        // Parse the output party information
        let output_parties = parse_output_parties(args.output_parties)?;
        // Obtain the compiled context
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
        // Print the serialized and compiled context to stdout
        println!("{}", serde_json::to_string(&compiled_context).unwrap());
        Ok(())
    });
}
