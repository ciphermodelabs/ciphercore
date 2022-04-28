#![cfg_attr(feature = "nightly-features", feature(backtrace))]
#[macro_use]
extern crate ciphercore_base;

use ciphercore_base::custom_ops::CustomOperation;
use ciphercore_base::data_types::{scalar_type, UINT64};
use ciphercore_base::errors::Result;
use ciphercore_base::graphs::create_context;
use ciphercore_base::ops::min_max::{Max, Min};
use ciphercore_utils::execute_main::execute_main;

fn main() {
    env_logger::init();
    execute_main(|| -> Result<()> {
        let c = create_context()?;
        let g = c.create_graph()?;
        let i1 = g.input(scalar_type(UINT64))?.a2b()?;
        let i2 = g.input(scalar_type(UINT64))?.a2b()?;
        let o = g.create_tuple(vec![
            g.custom_op(CustomOperation::new(Min {}), vec![i1.clone(), i2.clone()])?,
            g.custom_op(CustomOperation::new(Max {}), vec![i1, i2])?,
        ])?;
        g.set_output_node(o)?;
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
