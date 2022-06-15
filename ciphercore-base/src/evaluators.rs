pub mod get_result_util;
pub mod simple_evaluator;

use crate::data_values::Value;
use crate::errors::Result;
use crate::graphs::{Context, Operation};
use crate::graphs::{Graph, Node};
use crate::random::SEED_SIZE;

pub trait Evaluator {
    fn preprocess(&mut self, context: Context) -> Result<()> {
        context.check_finalized()?;
        for graph in context.get_graphs() {
            for node in graph.get_nodes() {
                node.get_type()?;
            }
        }
        Ok(())
    }

    fn evaluate_node(&mut self, node: Node, dependencies_values: Vec<Value>) -> Result<Value>;

    fn evaluate_call_iterate(
        &mut self,
        node: Node,
        dependencies_values: Vec<Value>,
    ) -> Result<Value> {
        match node.get_operation() {
            Operation::Call => {
                let graphs = node.get_graph_dependencies();
                self.evaluate_graph(graphs[0].clone(), dependencies_values)
            }
            Operation::Iterate => {
                let graphs = node.get_graph_dependencies();
                let initial_state_value = dependencies_values[0].clone();
                let inputs_value = dependencies_values[1].clone();
                let mut current_state_value = initial_state_value;
                let mut output_values = vec![];
                for input_value in inputs_value.to_vector()? {
                    let result = self.evaluate_graph(
                        graphs[0].clone(),
                        vec![current_state_value.clone(), input_value],
                    )?;
                    let result = result.to_vector()?;
                    current_state_value = result[0].clone();
                    output_values.push(result[1].clone());
                }
                Ok(Value::from_vector(vec![
                    current_state_value,
                    Value::from_vector(output_values),
                ]))
            }
            _ => {
                panic!("Should not be here!");
            }
        }
    }

    fn evaluate_graph(&mut self, graph: Graph, inputs_values: Vec<Value>) -> Result<Value> {
        graph.get_context().check_finalized()?;
        let mut num_input_nodes = 0;
        let nodes = graph.get_nodes();

        for node in nodes.iter() {
            if let Operation::Input(_) = node.get_operation() {
                num_input_nodes += 1;
            }
        }
        if num_input_nodes != inputs_values.len() {
            return Err(runtime_error!(
                "Incorrect number of inputs for evaluation: {} expected, but {} provided",
                num_input_nodes,
                inputs_values.len()
            ));
        }

        let mut node_option_values: Vec<Option<Value>> = vec![];

        let num_nodes = nodes.len();
        let mut to_consume_option = vec![0; num_nodes];
        for node in nodes.iter() {
            for dep in node.get_node_dependencies() {
                to_consume_option[dep.get_id() as usize] += 1;
            }
        }

        let output_node = graph.get_output_node()?;
        let output_id = output_node.get_id() as usize;
        let mut update_consumed_option_nodes = |node: Node, values: &mut [Option<Value>]| {
            for dep in node.get_node_dependencies() {
                let dep_id = dep.get_id() as usize;
                to_consume_option[dep_id] -= 1;
                if to_consume_option[dep_id] == 0 && dep_id != output_id {
                    values[dep_id] = None;
                }
            }
        };
        let mut input_id: u64 = 0;
        for node in nodes.iter() {
            let mut dependencies_values = vec![];
            for dependency in node.get_node_dependencies() {
                let node_value = node_option_values[dependency.get_id() as usize].clone();
                match node_value {
                    Some(value) => dependencies_values.push(value.clone()),
                    None => {
                        panic!("Dependency is already removed. Shouldn't be here.");
                    }
                }
            }
            match node.get_operation() {
                Operation::Input(t) => {
                    if !inputs_values[input_id as usize].check_type(t)? {
                        return Err(runtime_error!("Invalid input type"));
                    }
                    node_option_values.push(Some(inputs_values[input_id as usize].clone()));
                    input_id += 1;
                }
                Operation::Call | Operation::Iterate => {
                    let res = self.evaluate_call_iterate(node.clone(), dependencies_values)?;
                    node_option_values.push(Some(res));
                    update_consumed_option_nodes((*node).clone(), &mut node_option_values);
                }
                _ => {
                    let res = self.evaluate_node(node.clone(), dependencies_values)?;
                    node_option_values.push(Some(res.clone()));
                    update_consumed_option_nodes((*node).clone(), &mut node_option_values);
                }
            }
        }
        Ok(node_option_values[output_node.get_id() as usize]
            .clone()
            .unwrap())
    }

    fn evaluate_context(&mut self, context: Context, inputs_values: Vec<Value>) -> Result<Value> {
        context.check_finalized()?;
        self.evaluate_graph(context.get_main_graph()?, inputs_values)
    }
}

fn evaluate_simple_evaluator(
    graph: Graph,
    inputs: Vec<Value>,
    prng_seed: Option<[u8; SEED_SIZE]>,
) -> Result<Value> {
    let mut evaluator = simple_evaluator::SimpleEvaluator::new(prng_seed)?;
    evaluator.preprocess(graph.get_context())?;
    evaluator.evaluate_graph(graph, inputs)
}

/// Evaluate a given graph on a given set of inputs with a random PRNG seed.
pub fn random_evaluate(graph: Graph, inputs: Vec<Value>) -> Result<Value> {
    evaluate_simple_evaluator(graph, inputs, None)
}
