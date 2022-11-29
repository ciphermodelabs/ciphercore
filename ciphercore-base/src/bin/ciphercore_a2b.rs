//! Code example of a binary creating the serialization of a a2b context.
#![cfg_attr(feature = "nightly-features", feature(backtrace))]
extern crate ciphercore_base;

use std::str::FromStr;

use ciphercore_base::data_types::{array_type, ScalarType, BIT};
use ciphercore_base::errors::Result;
use ciphercore_base::graphs::{create_context, Context, Graph};

use ciphercore_utils::execute_main::execute_main;

use clap::Parser;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about=None)]
struct Args {
    #[clap(value_parser)]
    /// number of elements of an array (i.e., 2<sup>k</sup>)
    k: u32,
    /// scalar type of array elements
    #[clap(short, long, value_parser)]
    scalar_type: String,
    /// generate b2a graph instead of a2b
    #[clap(long, value_parser)]
    inverse: bool,
}

fn gen_a2b_graph(context: Context, k: u32, st: ScalarType) -> Result<Graph> {
    let graph = context.create_graph()?;
    let n = 2u64.pow(k);
    let i = graph.input(array_type(vec![n], st))?;
    graph.set_output_node(i.a2b()?)?;
    graph.finalize()?;
    Ok(graph)
}

fn gen_b2a_graph(context: Context, k: u32, st: ScalarType) -> Result<Graph> {
    let graph = context.create_graph()?;
    let n = 2u64.pow(k);
    let i = graph.input(array_type(vec![n, st.size_in_bits()], BIT))?;
    graph.set_output_node(i.b2a(st)?)?;
    graph.finalize()?;
    Ok(graph)
}

/// This binary prints the serialized a2b/b2a context on the non-encrypted input in (serde) JSON format.
///
/// It is needed for measuring performance of a2b/b2a conversions.
///
/// # Arguments
///
/// * `k` - number of elements of an array (i.e. 2<sup>n</sup>)
/// * `st` - scalar type of array elements
/// * `inverse` - generate b2a graph instead of a2b
///
/// # Usage
///
/// < this_binary > -s <st> <k>
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

        let graph = if args.inverse {
            gen_a2b_graph(context.clone(), args.k, st)
        } else {
            gen_b2a_graph(context.clone(), args.k, st)
        }?;

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
