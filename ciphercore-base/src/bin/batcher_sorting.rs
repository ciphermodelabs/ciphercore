#![cfg_attr(feature = "nightly-features", feature(backtrace))]
#[macro_use]
extern crate ciphercore_base;

use ciphercore_base::applications::sorting::create_batchers_sorting_graph;
use ciphercore_base::errors::Result;
use ciphercore_base::graphs::{create_context, Graph};
use std::env;

use ciphercore_utils::execute_main::execute_main;

/// - This binary prints the serialized sorting-graph context on the standard
/// output in (serde) JSON format.
/// - Main context graph is based on Batcher's algorithm.
/// - It requires two input arguments k and b:
///     k: Parameter for input size (2^{k}). This is the input to be sorted
/// by the graph.
///     b: Number of bits used for representing individual the array element
/// - Usage: ./<this_binary> <k> <b>
fn main() {
    // Initialize a logger that collects information about errors and panics within CipherCore.
    // This information can be accessed via RUST_LOG.
    env_logger::init();
    // Execute CipherCore code such that all the internal errors are properly formatted and logged.
    execute_main(|| -> Result<()> {
        let args: Vec<String> = env::args().collect();
        if args.len() != 3 {
            eprintln!("Usage:");
            eprintln!("{} <k> <b>", args[0]);
            eprintln!("where,\n\tk - Graph parameter specifying the number of elements, i.e. 2^{{k}}, to be sorted by the graph");
            eprintln!("\tb - Graph parameter specifying the number of bits required to represent the unsigned scalars within the graph\n\n");
            return Err(runtime_error!(
                "Invalid number of command-line arguments passed in for this binary"
            ));
        }
        let k: u32 = args[1].parse()?;
        let b: u64 = args[2].parse()?;
        let context = create_context()?;
        let graph: Graph = create_batchers_sorting_graph(context.clone(), k, b)?;
        context.set_main_graph(graph.clone())?;
        assert_eq!(graph, context.get_main_graph()?);
        context.finalize()?;
        println!("{}", serde_json::to_string(&context)?);
        Ok(())
    });
}
