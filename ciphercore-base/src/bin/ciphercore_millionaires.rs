//! Code example of a binary creating the serialization of a minimum context.
extern crate ciphercore_base;

use ciphercore_base::applications::millionaires::create_millionaires_graph;
use ciphercore_base::errors::Result;
use ciphercore_base::graphs::{create_context, Graph};

use ciphercore_utils::execute_main::execute_main;

/// This binary prints the serialized millionaires-problem-graph context on the non-encrypted input in (serde) JSON format.
///
/// # Usage
///
/// < this_binary >
fn main() {
    // Initialize a logger that collects information about errors and panics within CipherCore.
    // This information can be accessed via RUST_LOG.
    env_logger::init();
    // Execute CipherCore code such that all the internal errors are properly formatted and logged.
    execute_main(|| -> Result<()> {
        // Create a context
        let context = create_context()?;
        // Create a Millionaires' problem graph in the context
        let graph: Graph = create_millionaires_graph(context.clone())?;
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
