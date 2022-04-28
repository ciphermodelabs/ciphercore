#![cfg_attr(feature = "nightly-features", feature(backtrace))]
#[macro_use]
extern crate ciphercore_base;

use ciphercore_utils::execute_main::execute_main;

use ciphercore_base::data_types::{scalar_type, BIT};
use ciphercore_base::errors::Result;
use ciphercore_base::graphs::create_context;

fn main() {
    env_logger::init();
    execute_main(|| -> Result<()> {
        let c = create_context()?;
        let g = c.create_graph()?;
        let i = g.input(scalar_type(BIT))?;
        g.set_output_node(i)?;
        g.finalize()?;
        c.set_main_graph(g)?;
        c.finalize()?;
        println!(
            "{}",
            serde_json::to_string(&c).map_err(|_| runtime_error!("Can't serialize"))?
        );
        Ok(())
    });
}
