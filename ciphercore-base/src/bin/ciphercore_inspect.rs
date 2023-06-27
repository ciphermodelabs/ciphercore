//! Code of a binary printing statistics on a given serialized context.
extern crate ciphercore_base;

use ciphercore_base::data_types::{
    get_size_in_bits, Type, BIT, INT128, INT16, INT32, INT64, INT8, UINT128, UINT16, UINT32,
    UINT64, UINT8,
};
use ciphercore_base::errors::Result;
use ciphercore_base::evaluators::simple_evaluator::SimpleEvaluator;
use ciphercore_base::graphs::{Context, Graph, Node, NodeAnnotation, Operation};
use ciphercore_base::inline::inline_common::DepthOptimizationLevel;
use ciphercore_base::inline::inline_ops::{InlineConfig, InlineMode};
use ciphercore_base::mpc::mpc_compiler::prepare_context;
use ciphercore_utils::eprintln_or_log;
use ciphercore_utils::execute_main::execute_main;
use std::fs;

use std::collections::HashMap;

use clap::{ArgAction, ArgEnum, Parser};

struct InputInfo {
    name: String,
    type_string: String,
}
fn is_network_node(node: Node) -> Result<bool> {
    if let Operation::NOP = node.get_operation() {
        for annotation in node.get_annotations()? {
            if let NodeAnnotation::Send(_, _) = annotation {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn get_buffer_length_in_items(t: Type) -> u64 {
    if t.is_array() {
        t.get_shape().iter().product()
    } else {
        1
    }
}

fn calculate_integer_operations(node: Node) -> Result<u64> {
    match node.get_operation() {
        Operation::Add | Operation::Subtract | Operation::Multiply => {
            Ok(get_buffer_length_in_items(node.get_type()?))
        }
        Operation::Truncate(_) | Operation::Sum(_) | Operation::PRF(_, _) => {
            let dependency = node.get_node_dependencies()[0].clone();
            let inp_t = dependency.get_type()?;
            Ok(get_buffer_length_in_items(inp_t))
        }
        Operation::Random(t) => {
            // For random and PRF this is a very rough estimate.
            // The actual amount of calculations depends on the
            // implementation of the third-party crypto library used for them
            Ok(get_buffer_length_in_items(t))
        }
        Operation::Dot | Operation::Matmul => {
            // For Matrix Multiplication this is a very rough estimate.
            // The actual amount of calculations depends on the
            // implementation of the third-party linear algebra library used
            // Following calculations are based on the simple_evaluator evaluations
            // And as an approximation some optimizations for rank-1 matrices are ignored
            let dependency0 = node.get_node_dependencies()[0].clone();
            let dependency1 = node.get_node_dependencies()[1].clone();
            let type0 = dependency0.get_type()?;
            let type1 = dependency1.get_type()?;
            let shape0 = type0.get_shape();
            let shape1 = type1.get_shape();
            let result_len = get_buffer_length_in_items(node.get_type()?);
            if shape0.len() == 1 && shape1.len() == 1 {
                Ok(shape0[0])
            } else {
                let middle_dim = if shape1.len() > 1 {
                    shape1[shape1.len() - 2]
                } else {
                    shape1[0]
                };
                Ok(middle_dim * result_len)
            }
        }
        _ => Ok(1),
    }
}
fn calculate_network_rounds(graph: Graph) -> Result<u32> {
    let nodes = graph.get_nodes();
    let mut nops = HashMap::<Node, u32>::new();
    for node in nodes {
        let node_is_network = u32::from(is_network_node(node.clone())?);
        let dependee_nops_max = node
            .get_node_dependencies()
            .iter()
            .map(|n| *nops.get(n).unwrap_or(&0))
            .max()
            .unwrap_or(0);
        nops.insert(node, dependee_nops_max + node_is_network);
    }
    Ok(*nops.values().max().unwrap())
}

const TWO: u64 = 2;
fn format_bits(t_in_bits: u64) -> String {
    let t_in_bytes = t_in_bits / 8;
    if t_in_bytes >= TWO.pow(30) {
        format!("{:.2}GB", t_in_bytes as f32 / TWO.pow(30) as f32)
    } else if t_in_bytes >= TWO.pow(20) {
        format!("{:.2}MB", t_in_bytes as f32 / TWO.pow(20) as f32)
    } else if t_in_bytes >= TWO.pow(10) {
        format!("{:.2}KB", t_in_bytes as f32 / TWO.pow(10) as f32)
    } else {
        format!("{t_in_bytes}B")
    }
}

const TEN: u64 = 10;
fn format_operations(ops: u64) -> String {
    if ops >= TEN.pow(9) {
        format!("{:.2}G Ops", ops as f32 / TEN.pow(9) as f32)
    } else if ops >= TEN.pow(6) {
        format!("{:.2}M Ops", ops as f32 / TEN.pow(6) as f32)
    } else if ops >= TEN.pow(3) {
        format!("{:.2}K Ops", ops as f32 / TEN.pow(3) as f32)
    } else {
        format!("{ops} Ops")
    }
}

struct RamStats {
    peak_ram_bits: u64,
    total_ram_bits: u64,
}

fn calculate_ram_stats(graph: Graph) -> Result<RamStats> {
    let nodes = graph.get_nodes();
    let mut node_size = vec![];
    for node in nodes.iter() {
        node_size.push(get_size_in_bits(node.get_type()?)?);
    }
    let mut remaining_dependents = vec![0; nodes.len()];
    for node in nodes.iter() {
        for dep in node.get_node_dependencies() {
            remaining_dependents[dep.get_id() as usize] += 1;
        }
    }
    let mut max_ram = 0;
    let mut cur_ram = 0;
    let mut total_ram = 0;
    for node in nodes.iter() {
        cur_ram += node_size[node.get_id() as usize];
        total_ram += node_size[node.get_id() as usize];
        for dep in node.get_node_dependencies() {
            let dep_id = dep.get_id() as usize;
            remaining_dependents[dep_id] -= 1;
            if remaining_dependents[dep_id] == 0 {
                cur_ram -= node_size[dep_id];
            }
        }
        max_ram = max_ram.max(cur_ram);
    }
    Ok(RamStats {
        peak_ram_bits: max_ram,
        total_ram_bits: total_ram,
    })
}

pub(crate) fn print_stats(graph: Graph) -> Result<()> {
    let mut cnt = HashMap::<String, u64>::new();
    let mut inputs = Vec::<InputInfo>::new();
    let mut network_traffic_in_bits = 0;
    let mut total_integer_operations = 0;
    let mut total_bit_operations = 0;
    let mut total_8bits_operations = 0;
    let mut total_16bits_operations = 0;
    let mut total_32bits_operations = 0;
    let mut total_64bits_operations = 0;
    let mut total_128bits_operations = 0;
    for node in graph.get_nodes() {
        let op = node.get_operation();
        let op_name = format!("{op}");
        *cnt.entry(op_name).or_insert(0) += 1;
        match op {
            Operation::Input(_) => {
                let input = InputInfo {
                    name: node.get_name()?.unwrap_or_else(|| "unnamed".to_owned()),
                    type_string: format!("{}", node.get_type()?),
                };
                inputs.push(input);
            }
            Operation::NOP => {
                if is_network_node(node.clone())? {
                    network_traffic_in_bits += get_size_in_bits(node.get_type()?)?;
                }
            }
            Operation::Add
            | Operation::Subtract
            | Operation::Multiply
            | Operation::Dot
            | Operation::Matmul
            | Operation::Truncate(_)
            | Operation::Sum(_) => {
                let st = node.get_type()?.get_scalar_type();
                let ops = calculate_integer_operations(node.clone())?;
                match st {
                    BIT => total_bit_operations += ops,
                    UINT8 | INT8 => total_8bits_operations += ops,
                    UINT16 | INT16 => total_16bits_operations += ops,
                    UINT32 | INT32 => total_32bits_operations += ops,
                    UINT64 | INT64 => total_64bits_operations += ops,
                    UINT128 | INT128 => total_128bits_operations += ops,
                };
                total_integer_operations += ops;
            }
            _ => {}
        }
    }
    let mut entries: Vec<(String, u64)> = cnt.iter().map(|e| (e.0.clone(), *e.1)).collect();
    let network_rounds = calculate_network_rounds(graph.clone())?;

    entries.sort_by_key(|e| -(e.1 as i64));
    println!("-------Stats--------");
    println!("Inputs: ",);
    for (i, input) in inputs.iter().enumerate() {
        println!("  {}. Name:{}", i + 1, input.name);
        println!("  Type:{}", input.type_string);
    }

    let output_node = graph.get_output_node()?;
    let output_name = output_node
        .get_name()?
        .unwrap_or_else(|| "unnamed".to_owned());
    let output_type = format!("{}", output_node.get_type()?);
    println!("Output: ",);
    println!("  Name:{output_name}");
    println!("  Type:{output_type}");

    println!("Network rounds: {network_rounds}");
    println!("Network traffic: {}", format_bits(network_traffic_in_bits));
    let ram_stats = calculate_ram_stats(graph.clone())?;
    println!("Peak RAM: {}", format_bits(ram_stats.peak_ram_bits));
    println!("Total RAM: {}", format_bits(ram_stats.total_ram_bits));
    println!(
        "Total number of integer arithmetic operations:   {}",
        format_operations(total_integer_operations)
    );
    println!(
        "  Total number of 1-bit arithmetic operations:   {}",
        format_operations(total_bit_operations)
    );
    println!(
        "  Total number of 8-bit arithmetic operations:   {}",
        format_operations(total_8bits_operations)
    );
    println!(
        "  Total number of 16-bit arithmetic operations:  {}",
        format_operations(total_16bits_operations)
    );
    println!(
        "  Total number of 32-bit arithmetic operations:  {}",
        format_operations(total_32bits_operations)
    );
    println!(
        "  Total number of 64-bit arithmetic operations:  {}",
        format_operations(total_64bits_operations)
    );
    println!(
        "  Total number of 128-bit arithmetic operations:  {}",
        format_operations(total_128bits_operations)
    );
    println!("Total operations: {}", graph.get_nodes().len());
    println!("Operations: ",);
    for e in entries {
        println!("  {:<10}\t{}", e.0, e.1);
    }
    Ok(())
}

#[derive(Debug, ArgEnum, Clone)]
enum InlineModeArg {
    Simple,
    DepthOptimizedDefault,
    DepthOptimizedExtreme,
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about=None)]
struct Args {
    #[clap(value_parser)]
    /// Path to a file with a serialized context
    input_path: String,
    /// Optional flag to inline, instantiate custom operations and optimize graphs within a given context
    #[clap(action = ArgAction::SetTrue)]
    prepare: bool,
    #[clap(arg_enum, value_parser)]
    /// Mode of inlining that unrolls operation nodes in graphs.
    /// Possible values are `simple`, `depth-optimized-default`, `depth-optimized-extreme`.
    /// The default value is simple.
    inline_mode: Option<InlineModeArg>,
}

fn get_evaluator() -> Result<SimpleEvaluator> {
    SimpleEvaluator::new(None)
}

fn get_inline_mode(mode_val: Option<InlineModeArg>) -> InlineMode {
    match mode_val {
        Some(mode) => match mode {
            InlineModeArg::Simple => InlineMode::Simple,
            InlineModeArg::DepthOptimizedDefault => {
                InlineMode::DepthOptimized(DepthOptimizationLevel::Default)
            }
            InlineModeArg::DepthOptimizedExtreme => {
                InlineMode::DepthOptimized(DepthOptimizationLevel::Extreme)
            }
        },
        None => InlineMode::Simple,
    }
}

/// This binary prints statistics of a given serialized context.
///
/// # Arguments
///
/// * `input_path` - path to a serialized context
/// * `prepare` - (optional) flag to inline, instantiate custom operations and optimize graphs in a given context
/// * `inline_mode` - (optional) mode of inlining that unrolls operation nodes in graphs.
///    Possible values are `simple`, `depth-optimized-default`, `depth-optimized-extreme`.
///    The default value is `simple`.
///
/// # Usage
///
/// `< this_binary > <input_path> <prepare> <inline_mode>`
fn main() {
    // Initialize a logger that collects information about errors and panics within CipherCore.
    // This information can be accessed via RUST_LOG.
    env_logger::init();
    // Execute CipherCore code such that all the internal errors are properly formatted and logged.
    execute_main(|| -> Result<()> {
        let args = Args::parse();
        let buffer = fs::read_to_string(&args.input_path)?;
        let context = serde_json::from_str::<Context>(&buffer).unwrap();
        let context2 = if args.prepare {
            let evaluator0 = get_evaluator()?;
            prepare_context(
                context,
                InlineConfig {
                    default_mode: get_inline_mode(args.inline_mode),
                    ..Default::default()
                },
                evaluator0,
                false,
            )?
            .get_context()
        } else {
            context
        };

        eprintln_or_log!("Calculating stats...");
        print_stats(context2.get_main_graph()?)?;
        Ok(())
    });
}
