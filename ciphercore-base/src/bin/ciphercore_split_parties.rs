//! Code of a binary that takes typed inputs and splits them between the parties
#![cfg_attr(feature = "nightly-features", feature(backtrace))]
#[macro_use]
extern crate ciphercore_base;

use ciphercore_base::data_values::Value;
use ciphercore_base::errors::Result;
use ciphercore_base::mpc::mpc_compiler::IOStatus;
use ciphercore_base::random::PRNG;
use ciphercore_base::typed_value::TypedValue;
use ciphercore_utils::execute_main::execute_main;

use clap::Parser;
use std::fs;

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

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about=None)]
struct Args {
    #[clap(value_parser)]
    /// Path to file that contains the inputs
    inputs_path: String,
    #[clap(value_parser)]
    /// String comprising comma separated list of input parties' IDs, valid ID values include `0`, `1`, `2` OR `public` OR `secret-shared`.
    input_parties: String,
    #[clap(value_parser)]
    /// Path to the output file that contains the inputs prepared for party `0`
    parties_inputs_0: String,
    #[clap(value_parser)]
    /// Path to the output file that contains the inputs prepared for party `1`
    parties_inputs_1: String,
    #[clap(value_parser)]
    /// Path to the output file that contains the inputs prepared for party `2`
    parties_inputs_2: String,
}

/// This binary reads the inputs from the file and splits them (+ possibly secret-shares) between the parties according to a command line parameter.
///
/// # Arguments
///
/// * `inputs_path` - path to file that contains the inputs
/// * `input_parties` - string comprising of either a comma separated list of input parties' IDs, valid ID values include `0`, `1`, `2` OR `public` OR `secret-shared`
/// * `inputs_path_0` - path to the output file that contains the inputs prepared for party `0`
/// * `inputs_path_1` - path to the output file that contains the inputs prepared for party `1`
/// * `inputs_path_2` - path to the output file that contains the inputs prepared for party `2`
///
/// # Usage
///
/// < this_binary > <context_path> <inline_mode> <input_parties> <output_parties>
fn main() {
    env_logger::init();
    execute_main(|| -> Result<()> {
        let args = Args::parse();
        let json_inputs = fs::read_to_string(&args.inputs_path)?;
        let inputs = serde_json::from_str::<Vec<TypedValue>>(&json_inputs)?;
        let input_parties = get_tokens(args.input_parties)?;
        let mut split_inputs = vec![vec![], vec![], vec![]];
        if inputs.len() != input_parties.len() {
            return Err(runtime_error!(
                "Invalid number of inputs parties: {} expected, but {} found",
                inputs.len(),
                input_parties.len()
            ));
        }
        let mut prng = PRNG::new(None)?;
        for i in 0..inputs.len() {
            match input_parties[i] {
                IOStatus::Party(p) => {
                    for (j, result_item) in split_inputs.iter_mut().enumerate() {
                        if j as u64 == p {
                            result_item.push(inputs[i].clone());
                        } else {
                            result_item.push(TypedValue::new(
                                inputs[i].t.clone(),
                                Value::zero_of_type(inputs[i].t.clone()),
                            )?);
                        }
                    }
                }
                IOStatus::Public => {
                    for result_item in split_inputs.iter_mut() {
                        result_item.push(inputs[i].clone());
                    }
                }
                IOStatus::Shared => {
                    let parties_shares = inputs[i].get_local_shares_for_each_party(&mut prng)?;
                    for j in 0..3 {
                        split_inputs[j].push(parties_shares[j].clone());
                    }
                }
            }
        }
        fs::write(
            args.parties_inputs_0,
            serde_json::to_string(&split_inputs[0])?,
        )?;
        fs::write(
            args.parties_inputs_1,
            serde_json::to_string(&split_inputs[1])?,
        )?;
        fs::write(
            args.parties_inputs_2,
            serde_json::to_string(&split_inputs[2])?,
        )?;
        Ok(())
    });
}
