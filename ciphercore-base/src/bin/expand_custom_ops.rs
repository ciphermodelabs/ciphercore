use std::io::{self, Read};

use ciphercore_base::custom_ops::run_instantiation_pass;
use ciphercore_base::errors::Result;
use ciphercore_base::graphs::Context;
use ciphercore_utils::execute_main::execute_main;

fn main() {
    env_logger::init();
    execute_main(|| -> Result<()> {
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer)?;
        let c = serde_json::from_str::<Context>(&buffer)?;
        let mapped_c = run_instantiation_pass(c)?;
        println!("{}", serde_json::to_string(&mapped_c.get_context())?);
        Ok(())
    });
}
