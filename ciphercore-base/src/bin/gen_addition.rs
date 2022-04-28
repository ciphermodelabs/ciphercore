#[macro_use]
extern crate maplit;

use ciphercore_base::data_types::{scalar_type, INT32};
use ciphercore_base::data_values::Value;
use ciphercore_base::errors::Result;
use ciphercore_base::evaluators::random_evaluate;
use ciphercore_base::graphs::{create_context, Context};
use ciphercore_utils::execute_main::execute_main;

fn main() {
    env_logger::init();
    execute_main(|| -> Result<()> {
        let context = || -> Result<Context> {
            let context = create_context()?;
            let graph = context.create_graph()?.set_name("main")?;
            graph
                .input(scalar_type(INT32))?
                .set_name("a")?
                .add(graph.input(scalar_type(INT32))?.set_name("b")?)?
                .set_as_output()?;
            graph.finalize()?.set_as_main()?;
            context.finalize()?;
            Ok(context)
        }()?;
        let result = || -> Result<i32> {
            let g = context.retrieve_graph("main")?;
            let result = random_evaluate(
                g.clone(),
                g.prepare_input_values(hashmap! {
                    "a" => Value::from_scalar(123, INT32)?,
                    "b" => Value::from_scalar(654, INT32)?,
                })?,
            )?;
            let result = result.to_i32(INT32)?;
            Ok(result)
        }()?;
        assert_eq!(result, 777);
        println!("{}", serde_json::to_string(&context)?);
        Ok(())
    });
}
