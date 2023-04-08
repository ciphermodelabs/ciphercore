use crate::data_types::get_size_in_bits;
use crate::errors::Result;
use crate::graphs::{Graph, Node, NodeAnnotation, Operation};
use std::collections::HashSet;

use super::mpc_compiler::{get_zero_shares, recursively_sum_shares, PARTIES};

// Contains information on what nodes should be converted from 3-out-of-3 to 2-out-of-3 shares, i.e. reshared
struct ResharingConfig {
    // Nodes that need resharing after their operation
    pub nodes_to_reshare: HashSet<Node>,
    // Nodes containing 3-out-of-3 shares
    unreshared_nodes: HashSet<Node>,
}

impl ResharingConfig {
    fn new() -> Self {
        ResharingConfig {
            nodes_to_reshare: HashSet::new(),
            unreshared_nodes: HashSet::new(),
        }
    }

    // Local operation should propagate 3-out-of-3 sharing if at least one of the inputs is 3-out-of-3
    fn local_operation_handler(&mut self, node: Node) -> Result<()> {
        let dependencies = node.get_node_dependencies();

        if node.get_operation().is_broadcasting_called() {
            // These operations might broadcast inputs.
            // To avoid a blowup of PRFs, we compare the sizes of the input data to be reshared and the output data.
            // If output data is larger, we perform resharing of inputs.

            // Compute the size of unreshared data
            let mut unreshared_input_size = 0;

            for dep in &dependencies {
                if self.unreshared_nodes.contains(dep) {
                    unreshared_input_size += get_size_in_bits(dep.get_type()?)?;
                }
            }

            let output_size = get_size_in_bits(node.get_type()?)?;

            if output_size > unreshared_input_size {
                self.ensure_dependencies_are_reshared(&node);
            } else if unreshared_input_size > 0 {
                // This means that there is at least one 3-out-of-3 input.
                // Thus, output should be 3-out-of-3 as well.
                self.unreshared_nodes.insert(node);
            }
        } else {
            let mut is_one_input_unreshared = false;
            for dep in &dependencies {
                if self.unreshared_nodes.contains(dep) {
                    is_one_input_unreshared = true;
                }
            }

            if is_one_input_unreshared {
                self.unreshared_nodes.insert(node);
            }
        }

        Ok(())
    }

    fn ensure_dependencies_are_reshared(&mut self, node: &Node) {
        for dep_node in node.get_node_dependencies() {
            if self.unreshared_nodes.contains(&dep_node) {
                // Note that this affects other nodes with this dependency
                // Possible mistakes are fixed later in sanity_check.
                self.unreshared_nodes.remove(&dep_node);
                self.nodes_to_reshare.insert(dep_node);
            }
        }
    }

    // Computes the set of nodes that need resharing of their output
    fn compute_graph_resharing(
        &mut self,
        graph: &Graph,
        private_nodes: &HashSet<Node>,
    ) -> Result<()> {
        self.nodes_to_reshare = HashSet::new();
        // nodes that contain 3-out-of-3 shares
        // All the other shared nodes have 2-out-of-3 shares
        self.unreshared_nodes = HashSet::new();

        for node in graph.get_nodes() {
            // if node is public, no resharing is needed
            if !private_nodes.contains(&node) {
                continue;
            }
            let op = node.get_operation();
            if !op.is_mpc_compiled() {
                return Err(runtime_error!("This operation shouldn't be MPC compiled"));
            }
            match op {
                Operation::Input(_)
                 => {
                    // No resharing is needed
                }
                Operation::Add
                | Operation::Subtract
                | Operation::Sum(_)
                | Operation::CumSum(_)
                | Operation::Get(_)
                | Operation::Stack(_)
                | Operation::Concatenate(_)
                | Operation::Reshape(_)
                | Operation::PermuteAxes(_)
                | Operation::Zip
                | Operation::Repeat(_)
                | Operation::TupleGet(_)
                | Operation::CreateNamedTuple(_)
                | Operation::NamedTupleGet(_)
                | Operation::VectorToArray
                | Operation::VectorGet
                | Operation::CreateTuple
                | Operation::ArrayToVector
                | Operation::CreateVector(_) => {
                    self.local_operation_handler(node)?;
                }
                Operation::Multiply
                | Operation::Dot
                | Operation::Matmul
                | Operation::Gemm(_, _) => {
                    let dependencies = node.get_node_dependencies();

                    let mut all_inputs_are_shared = true;
                    for dep_node in &dependencies {
                        if !private_nodes.contains(dep_node) {
                            all_inputs_are_shared = false;
                        }
                    }

                    if all_inputs_are_shared {
                        // All inputs are shared.
                        // Then, the operation protocol needs all the input shares being 2-out-of-3.
                        self.ensure_dependencies_are_reshared(&node);
                        self.unreshared_nodes.insert(node);
                    } else {
                        // If at least one of the inputs is public, operation is local
                        self.local_operation_handler(node)?;
                    }
                }
                Operation::Join(_, _)
                | Operation::Truncate(_)
                | Operation::A2B
                | Operation::B2A(_)
                | Operation::Sort(_)
                // empirical observation that our graphs usually have several GetSlice nodes from one node; so it's better to reshare its input once.
                | Operation::GetSlice(_)
                 => {
                    self.ensure_dependencies_are_reshared(&node);
                }
                Operation::MixedMultiply | Operation::ApplyPermutation(_) => {
                    let dependencies = node.get_node_dependencies();

                    if private_nodes.contains(&dependencies[1]) {
                        // If bits/permutation are shared, MixedMultiply/ApplyPermutation needs only 2-out-of-3 inputs
                        self.ensure_dependencies_are_reshared(&node);
                    } else {
                        // If bits/permutation are public, MixedMultiply/ApplyPermutation is local
                        self.local_operation_handler(node)?;
                    }
                },
                _ => {
                    return Err(runtime_error!("Unrecognized operation {}", op));
                }
            }
        }

        // Check that the output node is reshared
        let out_node = graph.get_output_node()?;
        if self.unreshared_nodes.contains(&out_node) {
            self.nodes_to_reshare.insert(out_node);
        }

        self.sanity_pass(graph, private_nodes)
    }

    // Check that nodes marked as "to be reshared" have input 3-out-of-3 nodes.
    // If not, they don't need resharing.
    fn sanity_pass(&mut self, graph: &Graph, shared_nodes: &HashSet<Node>) -> Result<()> {
        for node in graph.get_nodes() {
            match node.get_operation() {
                Operation::Multiply
                | Operation::Matmul
                | Operation::Dot
                | Operation::Gemm(_, _) => {
                    let dependencies = node.get_node_dependencies();

                    let mut all_inputs_are_shared = true;
                    for dep_node in &dependencies {
                        if !shared_nodes.contains(dep_node) {
                            all_inputs_are_shared = false;
                        }
                    }

                    if all_inputs_are_shared {
                        continue;
                    }
                }
                _ => {}
            }
            if self.nodes_to_reshare.contains(&node) {
                let mut node_should_be_reshared = false;
                for dep in node.get_node_dependencies() {
                    if self.unreshared_nodes.contains(&dep) {
                        node_should_be_reshared = true;
                    }
                }
                if !node_should_be_reshared {
                    self.nodes_to_reshare.remove(&node);
                }
            }
            if self.unreshared_nodes.contains(&node) {
                let mut node_is_unreshared = false;
                for dep in node.get_node_dependencies() {
                    if self.unreshared_nodes.contains(&dep) {
                        node_is_unreshared = true;
                    }
                }
                if !node_is_unreshared {
                    self.unreshared_nodes.remove(&node);
                }
            }
        }

        Ok(())
    }
}

pub(super) fn get_nodes_to_reshare(
    graph: &Graph,
    shared_nodes: &HashSet<Node>,
) -> Result<HashSet<Node>> {
    let mut resharing_config = ResharingConfig::new();
    resharing_config.compute_graph_resharing(graph, shared_nodes)?;
    Ok(resharing_config.nodes_to_reshare)
}

// Convert 3-out-of-3 to 2-out-of-3 shares
pub(super) fn reshare(input_shares: &Node, prf_keys: &Node) -> Result<Node> {
    let g = input_shares.get_graph();

    let input_shares_vec: Vec<Node> = (0..PARTIES)
        .map(|i| input_shares.tuple_get(i as u64).unwrap())
        .collect();
    let zero_shares =
        get_zero_shares(g.clone(), prf_keys.clone(), input_shares_vec[0].get_type()?)?;

    let mut output_shares_vec = vec![];
    for i in 0..PARTIES {
        // Party i masks its 3-out-of-3 share: input_i + zero_share_i
        let masked_share = recursively_sum_shares(
            g.clone(),
            vec![input_shares_vec[i].clone(), zero_shares[i].clone()],
        )?;
        // Party i sends output share to party (i-1)
        let sent_share = g.nop(masked_share)?;
        let im1 = ((i + PARTIES - 1) % PARTIES) as u64;
        sent_share.add_annotation(NodeAnnotation::Send(i as u64, im1))?;
        output_shares_vec.push(sent_share);
    }
    g.create_tuple(output_shares_vec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_types::{array_type, BIT, INT32};
    use crate::graphs::create_context;
    use crate::mpc::mpc_compiler::propagate_private_annotations;

    #[test]
    fn test_resharing() -> Result<()> {
        {
            let c = create_context()?;
            let g = c.create_graph()?;
            let i1 = g.input(array_type(vec![2, 10], BIT))?;
            let i2 = g.input(array_type(vec![10, 3], BIT))?;
            let prod = i1.matmul(i2)?;
            let out = prod.sum(vec![0])?;
            out.set_as_output()?;
            g.finalize()?;
            g.set_as_main()?;
            c.finalize()?;

            let shared_nodes = propagate_private_annotations(g.clone(), vec![true, true])?.0;
            let reshared_nodes = get_nodes_to_reshare(&g, &shared_nodes)?;

            assert!(reshared_nodes.len() == 1);
            assert!(reshared_nodes.contains(&out));

            let shared_nodes = propagate_private_annotations(g.clone(), vec![false, true])?.0;
            let reshared_nodes = get_nodes_to_reshare(&g, &shared_nodes)?;

            assert!(reshared_nodes.len() == 0);

            let shared_nodes = propagate_private_annotations(g.clone(), vec![false, false])?.0;
            let reshared_nodes = get_nodes_to_reshare(&g, &shared_nodes)?;

            assert!(reshared_nodes.len() == 0);
        }

        {
            let c = create_context()?;
            let g = c.create_graph()?;
            let i1 = g.input(array_type(vec![2, 10], BIT))?;
            let i2 = g.input(array_type(vec![10, 3], BIT))?;
            let prod = i1.matmul(i2)?;
            let i3 = g.input(array_type(vec![3, 4], BIT))?;
            let out = prod.matmul(i3)?;
            out.set_as_output()?;
            g.finalize()?;
            g.set_as_main()?;
            c.finalize()?;

            let shared_nodes = propagate_private_annotations(g.clone(), vec![true, true, true])?.0;
            let reshared_nodes = get_nodes_to_reshare(&g, &shared_nodes)?;

            assert!(reshared_nodes.len() == 2);
            assert!(reshared_nodes.contains(&prod));
            assert!(reshared_nodes.contains(&out));

            let shared_nodes = propagate_private_annotations(g.clone(), vec![false, true, true])?.0;
            let reshared_nodes = get_nodes_to_reshare(&g, &shared_nodes)?;

            assert!(reshared_nodes.len() == 1);
            assert!(reshared_nodes.contains(&out));

            let shared_nodes = propagate_private_annotations(g.clone(), vec![true, true, false])?.0;
            let reshared_nodes = get_nodes_to_reshare(&g, &shared_nodes)?;

            assert!(reshared_nodes.len() == 1);
            assert!(reshared_nodes.contains(&prod));
        }
        {
            let c = create_context()?;
            let g = c.create_graph()?;
            let i1 = g.input(array_type(vec![2, 10], BIT))?;
            let i2 = g.input(array_type(vec![10, 3], BIT))?;
            let prod12 = i1.matmul(i2)?;
            let i3 = g.input(array_type(vec![2, 4], BIT))?;
            let i4 = g.input(array_type(vec![4, 3], BIT))?;
            let prod34 = i3.matmul(i4)?;
            let out = prod12.add(prod34)?;
            out.set_as_output()?;
            g.finalize()?;
            g.set_as_main()?;
            c.finalize()?;

            let shared_nodes = propagate_private_annotations(g.clone(), vec![true; 4])?.0;
            let reshared_nodes = get_nodes_to_reshare(&g, &shared_nodes)?;

            assert!(reshared_nodes.len() == 1);
            assert!(reshared_nodes.contains(&out));
        }
        {
            let c = create_context()?;
            let g = c.create_graph()?;
            let i1 = g.input(array_type(vec![2, 10], INT32))?;
            let i2 = g.input(array_type(vec![10, 3], INT32))?;
            let prod12 = i1.matmul(i2)?;
            let i3 = g.input(array_type(vec![2, 3], INT32))?;
            let s123 = prod12.add(i3)?;
            let out = s123.a2b()?;
            out.set_as_output()?;
            g.finalize()?;
            g.set_as_main()?;
            c.finalize()?;

            let shared_nodes = propagate_private_annotations(g.clone(), vec![true; 3])?.0;
            let reshared_nodes = get_nodes_to_reshare(&g, &shared_nodes)?;

            assert!(reshared_nodes.len() == 1);
            assert!(reshared_nodes.contains(&s123));
        }

        Ok(())
    }
}
