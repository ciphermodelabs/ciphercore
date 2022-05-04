use ciphercore_base::errors::Result;
use ciphercore_base::graphs::*;
use clap::Parser;
use std::fs;

fn get_graphviz_node_ref(node: Node) -> String {
    return format!(
        "{}{}_{}",
        node.get_operation(),
        node.get_id(),
        node.get_graph().get_id()
    );
}

fn get_graphviz_graph_ref(graph: Graph) -> String {
    return format!("cluster{}", graph.get_id());
}

// Returns the sender and receiver party information, given an `NOP` node,
// formatted as:
// `Send[sender_id->receiver_id]`
// Assuming that the compiled graph would have a single `Send(_, _)` annotation.
fn get_send_anno_info_str(node: Node) -> Result<String> {
    let mut send_info_str = String::from("");
    if Operation::NOP == node.get_operation() {
        let mut send_count = 0;
        let node_annotations = node.get_annotations()?;
        for annotation in node_annotations {
            if let NodeAnnotation::Send(sender_id, receiver_id) = annotation {
                let comma_req = if send_count != 0 {
                    ", "
                } else {
                    // Atleast one `Send(_, _)` annotation present, create a new
                    // line in preparation for display
                    send_info_str.push_str("\\n");
                    // First `Send(_, _)`, therefore no comma required
                    ""
                };
                send_info_str.push_str(&format!(
                    "{}Send[{}->{}]",
                    comma_req, sender_id, receiver_id
                ));
                send_count += 1;
            }
        }
    }
    Ok(send_info_str)
}

fn get_graphviz_node_def(node: Node) -> Result<String> {
    let mut node_color = String::from("");
    let mut is_input_node: bool = false;
    let mut is_output_node: bool = false;
    let mut type_str = String::from("");

    let node_operation = node.get_operation();
    if let Operation::Input(_) = node_operation {
        is_input_node = true;
    }
    if node == node.get_graph().get_output_node()? {
        is_output_node = true;
    }
    if is_input_node && is_output_node {
        node_color.push_str(", shape=box, style=filled, color=magenta");
    } else if is_input_node {
        node_color.push_str(", shape=box, style=filled, color=royalblue");
    } else if is_output_node {
        node_color.push_str(", shape=box, style=filled, color=crimson");
    }
    // Generate a string according to the node type
    let node_type = node.get_type()?;
    type_str.push_str(&(node_type.to_string()));
    let node_name = if let Ok(s) = node.get_name() {
        format!("\\n{}\\n", s)
    } else {
        "".to_owned()
    };
    // Get annotation information string
    let send_ann_str = get_send_anno_info_str(node.clone())?;
    Ok(format!(
        "\t{} [label=\"{}{}\\n{}{}\"{}]\n",
        get_graphviz_node_ref(node),
        node_operation,
        send_ann_str,
        type_str,
        node_name,
        node_color
    ))
}

fn get_graphviz_edge_def(dependent_node: Node, dependee_node: Node) -> String {
    return format!(
        "\t\t{} -> {}\n",
        get_graphviz_node_ref(dependent_node),
        get_graphviz_node_ref(dependee_node)
    );
}

fn get_graphviz_open_subgraph(graph: Graph) -> String {
    let graph_name = if let Ok(s) = graph.get_name() {
        format!("label = \"{}\"\n", s)
    } else {
        "label = \"\"\n".to_string()
    };
    return format!(
        "\n\tsubgraph {} {{\n{}",
        get_graphviz_graph_ref(graph),
        graph_name
    );
}

fn get_graphviz_close_subgraph() -> &'static str {
    "\t}\n"
}

fn graphviz_open_graph() -> &'static str {
    "digraph{\n"
}

fn graphviz_close_graph() -> &'static str {
    "}"
}

/// Function to add edges from the node to the cluster/subgraph referred to by
/// the dependency_graph. Number of edges drawn corresponds to the number of
/// input nodes for the dependency_graph
fn add_edges_from_node_to_dependency_graph(
    node: Node,
    dependency_graph: Graph,
    viz_str: &mut String,
) {
    for input_node in dependency_graph.get_nodes() {
        if let Operation::Input(..) = input_node.get_operation() {
            let graph_node_edge = format!(
                "\t{} -> {} [lhead={}];\n",
                get_graphviz_node_ref(node.clone()),
                get_graphviz_node_ref(input_node.clone()),
                get_graphviz_graph_ref(dependency_graph.clone())
            );
            viz_str.push_str(&graph_node_edge.clone());
        }
    }
}

/// Given a context, this function prints all the graphs and nodes within the
/// context.
/// Inputs:
///     1. context - Context for the graph
///     2. viz_str - &mut String that will store the DOT graphviz code, if function is successful
/// Assumption: This function depends on the nodes being topologically sorted
fn generate_graphviz_context_code(context: Context, viz_str: &mut String) -> Result<()> {
    // Open the context
    viz_str.push_str(graphviz_open_graph());
    let context_graphs = context.get_graphs();
    for graph in context_graphs {
        // Open subgraph: Open cluster for the graph
        viz_str.push_str(&get_graphviz_open_subgraph(graph.clone()));

        let graph_nodes = graph.clone().get_nodes();
        for node in graph_nodes {
            // Add node definition
            viz_str.push_str(&get_graphviz_node_def(node.clone())?);
            // Now, add the dependency edges, dependency nodes must have already been created
            // since the nodes have been topologically sorted
            let node_dependencies = node.get_node_dependencies();
            for dependency in node_dependencies.clone() {
                viz_str.push_str(&get_graphviz_edge_def(dependency.clone(), node.clone()));
            }
            // Similarly, add dependency edges esp. for Operations like Call(), Iterate() etc.
            // Assuming that the graphs have also been topologically sorted
            let graph_dependencies = node.get_graph_dependencies();
            for dependency_graph in graph_dependencies {
                add_edges_from_node_to_dependency_graph(node.clone(), dependency_graph, viz_str);
            }
        }
        // Close subgraph: Close the cluster for the graph
        viz_str.push_str(get_graphviz_close_subgraph());
    }
    // Close the context
    viz_str.push_str(graphviz_close_graph());
    Ok(())
}

use ciphercore_utils::execute_main::execute_main;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about=None)]
struct Args {
    /// Path to file that contains the serialized Context
    context_path: String,
}

/// - Command to generate this binary:
///     cargo build --release
///     (This generates the binary `visualize_context` at ./target/release/visualize_context)
/// - Command to generate the graphviz graph-based context DOT code
///     ./<prog_to_print_some_string_of serialized_context> | ./target/release/visualize_context  > <context_dot_code_file_name>.gv
///     (.gv file contains the DOT-based graphviz code)
/// - Command to generate the diagram of the serialized context from the .gv file
///     For jpeg file:
///         dot -Tjpeg <context_dot_code_file_name>.gv -o sample_graph.jpeg
///     For png file:
///         dot -Tpng <context_dot_code_file_name>.gv -o sample_graph.png
///     For pdf file:
///         dot -Tpdf <context_dot_code_file_name>.gv -o sample_graph.pdf
///     (See manual entry for dot i.e. 'man dot' for more details)
fn main() {
    env_logger::init();
    execute_main(|| -> Result<()> {
        let args = Args::parse();
        let serialized_context = fs::read_to_string(&args.context_path)?;
        let context: Context = serde_json::from_str::<Context>(&serialized_context)?;
        let mut viz_code = String::from("");
        generate_graphviz_context_code(context, &mut viz_code)?;
        println!("{}", viz_code);
        Ok(())
    });
}
