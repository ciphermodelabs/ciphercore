use std::io::{self, Read};

use ciphercore_utils::execute_main::execute_main;

use ciphercore_base::errors::Result;
use ciphercore_base::graphs::Context;
use ciphercore_base::inline::inline_ops::{inline_operations, InlineConfig, InlineMode};

fn main() {
    env_logger::init();
    execute_main(|| -> Result<()> {
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer)?;
        let c = serde_json::from_str::<Context>(&buffer)?;
        let result_c = inline_operations(
            c,
            InlineConfig {
                default_mode: InlineMode::Simple,
                ..Default::default()
            },
        )?;
        println!("{}", serde_json::to_string(&result_c)?);
        Ok(())
    });
}
