//! Code example of a binary creating the serialization of a minimum context.
#![cfg_attr(feature = "nightly-features", feature(backtrace))]
extern crate ciphercore_base;

use std::str::FromStr;

use ciphercore_base::applications::minimum::create_minimum_graph;
use ciphercore_base::data_types::ScalarType;
use ciphercore_base::errors::Result;
use ciphercore_base::graphs::{create_context, Graph};

use ciphercore_utils::execute_main::execute_main;

use clap::Parser;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about=None)]
struct Args {
    #[clap(value_parser)]
    /// parameter defining the number of elements of an array (i.e. 2<sup>n</sup>)
    n: u64,
    /// scalar type of array elements
    #[clap(short, long, value_parser)]
    scalar_type: String,
}

/// This binary prints the serialized minimum-graph context on the non-encrypted input in (serde) JSON format.
///
/// # Arguments
///
/// * `n` - number of elements of an array (i.e. 2<sup>n</sup>)
/// * `st` - scalar type of matrix elements (choose `b` for binary entries or one of Rust integer types: `i8`, `u8`, ..., `i64`, `u64`)
///
/// # Usage
///
/// < this_binary > -s <st> <n>
fn main() {
    // Initialize a logger that collects information about errors and panics within CipherCore.
    // This information can be accessed via RUST_LOG.
    env_logger::init();
    // Execute CipherCore code such that all the internal errors are properly formatted and logged.
    execute_main(|| -> Result<()> {
        let args = Args::parse();
        let st = ScalarType::from_str(&args.scalar_type)?;
        // Create a context
        let context = create_context()?;
        // Create a minimum graph in the context
        let graph: Graph = create_minimum_graph(context.clone(), args.n, st)?;
        // Set this graph as main to be able to finalize the context
        context.set_main_graph(graph)?;
        // Finalize the context. This makes sure that all the graphs of the contexts are ready for computation.
        // After this action the context can't be changed.
        context.finalize()?;
        // Serialize the context and print it to stdout
        println!("{}", serde_json::to_string(&context)?);
        Ok(())
    });
}
