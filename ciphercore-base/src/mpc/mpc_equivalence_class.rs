use crate::data_types::{get_types_vector, Type};
use crate::errors::Result;
use crate::graphs::{Context, Node, NodeAnnotation, Operation};
use crate::mpc::mpc_compiler::IOStatus;
use std::collections::HashMap;
use std::sync::Arc;

//The Vec<Vec<u64>> represents which party(ies) holds the same value/share,
//e.g., [[0],[1],[2]] means each party holds a distinct share, [[0,1],[2]] means party 0 and party 1 are holding the same share while party 2 holds another one.
#[derive(Clone, Debug)]
pub(super) enum EquivalenceClasses {
    Atomic(Vec<Vec<u64>>),
    Vector(Vec<EquivalenceClassesPointer>),
}

type EquivalenceClassesPointer = Arc<EquivalenceClasses>;

impl PartialEq for EquivalenceClasses {
    /// Recursively checks equality of two EquivalenceClasses
    fn eq(&self, other: &Self) -> bool {
        match self {
            EquivalenceClasses::Atomic(d) => {
                if let EquivalenceClasses::Atomic(other_d) = other {
                    compare_vector_of_vector((*d).clone(), (*other_d).clone())
                        .expect("EquivalenceClasses comparison error!")
                } else {
                    false
                }
            }
            EquivalenceClasses::Vector(v) => {
                if let EquivalenceClasses::Vector(other_v) = other {
                    if v.len() != other_v.len() {
                        return false;
                    }
                    for i in 0..v.len() {
                        if v[i] != other_v[i] {
                            return false;
                        }
                    }
                    true
                } else {
                    false
                }
            }
        }
    }
}

fn compare_vector_of_vector(vector1: Vec<Vec<u64>>, vector2: Vec<Vec<u64>>) -> Result<bool> {
    if vector1.len() != vector2.len() {
        return Ok(false);
    }
    //Compare each element of vector1 with each element of vector2 until find a match. Use an bool vector to mark the matched elements in vector2. Return true if every element in vector2 has a match.
    let mut matched = vec![false; vector1.len()];
    let mut unmatched_count = vector1.len();
    for sub_vec1 in vector1 {
        for (index, sub_vec2) in vector2.iter().enumerate() {
            if matched[index] {
                continue;
            }
            if compare_vector(sub_vec1.clone(), sub_vec2.to_vec())? {
                matched[index] = true;
                unmatched_count -= 1;
                continue;
            }
        }
    }
    Ok(unmatched_count == 0)
}

fn compare_vector(vector1: Vec<u64>, vector2: Vec<u64>) -> Result<bool> {
    if vector1.len() != vector2.len() {
        return Ok(false);
    }
    let mut matched = vec![false; vector1.len()];
    let mut unmatched_count = vector1.len();
    for element1 in vector1 {
        for (index, element2) in vector2.iter().enumerate() {
            if matched[index] {
                continue;
            }
            if element1 == *element2 {
                matched[index] = true;
                unmatched_count -= 1;
                continue;
            }
        }
    }
    Ok(unmatched_count == 0)
}

//This function generates equivalence classes for all nodes in the input compiled context.
//As the index of VectorGet depends on the input data, we cannot return a correct EquivalenceClasses without the knowledge of the input data. Thus, we restrict that all elements in the input Vector should have a same EquivalenceClasses.
#[allow(dead_code)]
pub(super) fn generate_equivalence_class(
    context: Context,
    input_party_map: Vec<Vec<IOStatus>>,
) -> Result<HashMap<(u64, u64), EquivalenceClasses>> {
    let mut equivalence_classes: HashMap<(u64, u64), EquivalenceClasses> = HashMap::new();
    let mut input_count = 0;
    let graphs = context.get_graphs();
    for graph in graphs {
        let nodes = graph.get_nodes();
        for node in nodes {
            let dependencies = node.get_node_dependencies();
            let mut dependencies_class = vec![];
            for dependence_node in &dependencies {
                dependencies_class.push(Arc::new(
                    equivalence_classes
                        .get(&dependence_node.get_global_id())
                        .expect("hashmap get error!")
                        .clone(),
                ));
            }
            match node.get_operation() {
                Operation::Input(input_type) => {
                    equivalence_classes.insert(
                        node.get_global_id(),
                        get_input_class(input_type, &input_party_map[0][input_count])?,
                    );
                    input_count += 1;
                }
                Operation::CreateTuple
                | Operation::CreateNamedTuple(_)
                | Operation::CreateVector(_) => {
                    equivalence_classes.insert(
                        node.get_global_id(),
                        EquivalenceClasses::Vector(dependencies_class),
                    );
                }

                Operation::TupleGet(field_id) => {
                    let previous_class = (*dependencies_class[0]).clone();
                    if let EquivalenceClasses::Vector(v) = previous_class {
                        let target_class = v[field_id as usize].clone();
                        equivalence_classes.insert(node.get_global_id(), (*target_class).clone());
                    }
                }

                Operation::NamedTupleGet(ref field_name) => {
                    let tuple_type = dependencies[0].get_type()?;
                    let mut field_id: Option<u64> = None;
                    if let Type::NamedTuple(ref v) = tuple_type {
                        for (id, (current_field_name, _)) in v.iter().enumerate() {
                            if current_field_name.eq(field_name) {
                                field_id = Some(id as u64);
                                break;
                            }
                        }
                    }
                    let field_id_raw = field_id.unwrap();
                    let namedtuple = (*dependencies_class[0]).clone();
                    if let EquivalenceClasses::Vector(v) = namedtuple {
                        equivalence_classes
                            .insert(node.get_global_id(), (*v[field_id_raw as usize]).clone());
                    }
                }

                Operation::Random(t) => {
                    equivalence_classes.insert(
                        node.get_global_id(),
                        recursive_class_filler(
                            t,
                            EquivalenceClasses::Atomic(vec![vec![0], vec![1], vec![2]]),
                        )?,
                    );
                }

                Operation::NOP => {
                    let mut previous_class = (*dependencies_class[0]).clone();
                    let annotation = context.get_node_annotations(node.clone())?;
                    for single_communication in annotation {
                        if let NodeAnnotation::Send(source_party, destination_party) =
                            single_communication
                        {
                            previous_class =
                                get_nop_class(previous_class, source_party, destination_party)?;
                        }
                    }
                    equivalence_classes.insert(node.get_global_id(), previous_class);
                }

                Operation::PRF(_, t) => {
                    equivalence_classes.insert(
                        node.get_global_id(),
                        recursive_class_filler(
                            t,
                            equivalence_classes
                                .get(&dependencies[0].get_global_id())
                                .expect("hashmap get error!")
                                .clone(),
                        )?,
                    );
                }

                Operation::Add
                | Operation::Subtract
                | Operation::Multiply
                | Operation::Dot
                | Operation::Matmul => {
                    equivalence_classes.insert(
                        node.get_global_id(),
                        combine_class(
                            (*dependencies_class[0]).clone(),
                            (*dependencies_class[1]).clone(),
                        )?,
                    );
                }

                Operation::Truncate(_)
                | Operation::Sum(_)
                | Operation::Get(_)
                | Operation::GetSlice(_)
                | Operation::Reshape(_)
                | Operation::A2B
                | Operation::B2A(_) => {
                    equivalence_classes
                        .insert(node.get_global_id(), (*dependencies_class[0]).clone());
                }

                Operation::PermuteAxes(_) => {
                    let previous_class = (*dependencies_class[0]).clone();
                    if let EquivalenceClasses::Atomic(d) = previous_class {
                        let current_class = d;
                        equivalence_classes.insert(
                            node.get_global_id(),
                            EquivalenceClasses::Atomic(current_class),
                        );
                    }
                }

                Operation::Stack(_) | Operation::VectorToArray => {
                    let previous_class = (*dependencies_class[0]).clone();
                    match previous_class {
                        EquivalenceClasses::Atomic(d) => {
                            let current_class = d;
                            equivalence_classes.insert(
                                node.get_global_id(),
                                EquivalenceClasses::Atomic(current_class),
                            );
                        }

                        EquivalenceClasses::Vector(v) => {
                            let mut current_class = (*v[0]).clone();
                            for e in v {
                                current_class = combine_class(current_class, (*e).clone())?;
                            }
                            equivalence_classes.insert(node.get_global_id(), current_class);
                        }
                    }
                }

                Operation::Zip => {
                    let mut current_class = vec![];
                    let mut index = 0;
                    'result_entries: loop {
                        let mut row = vec![];
                        for dependency_class in dependencies_class.clone() {
                            if let EquivalenceClasses::Vector(v) = &*dependency_class {
                                if v.len() <= index {
                                    break 'result_entries;
                                }
                                row.push(v[index].clone());
                            }
                        }
                        current_class.push(Arc::new(EquivalenceClasses::Vector(row)));
                        index += 1;
                    }
                    equivalence_classes.insert(
                        node.get_global_id(),
                        EquivalenceClasses::Vector(current_class),
                    );
                }

                Operation::Constant(t, _) => {
                    equivalence_classes.insert(
                        node.get_global_id(),
                        recursive_class_filler(t, EquivalenceClasses::Atomic(vec![vec![0, 1, 2]]))?,
                    );
                }

                Operation::Repeat(n) => {
                    let mut current_class = vec![];
                    for _ in 0..n {
                        current_class.push(Arc::new((*dependencies_class[0]).clone()));
                    }
                    equivalence_classes.insert(
                        node.get_global_id(),
                        EquivalenceClasses::Vector(current_class),
                    );
                }

                Operation::ArrayToVector => {
                    let mut current_class = vec![];

                    let dependency_node = dependencies[0].clone();
                    let shape = dependency_node.get_type()?.get_shape();
                    for _ in 0..shape[0] {
                        current_class.push(Arc::new((*dependencies_class[0]).clone()));
                    }
                    equivalence_classes.insert(
                        node.get_global_id(),
                        EquivalenceClasses::Vector(current_class),
                    );
                }

                Operation::VectorGet => {
                    let previous_class = (*dependencies_class[0]).clone();
                    if let EquivalenceClasses::Vector(v) = previous_class {
                        let current_class = (*v[0]).clone();
                        for class in v {
                            if current_class != *class {
                                panic!("elements in Vector have different EquivalenceClasses");
                            }
                        }
                        equivalence_classes.insert(node.get_global_id(), current_class);
                    }
                }

                _ => return Err(runtime_error!("node not supported")),
            }
        }
    }
    Ok(equivalence_classes)
}

fn get_input_class(t: Type, input_party: &IOStatus) -> Result<EquivalenceClasses> {
    match input_party {
        IOStatus::Public => Ok(recursive_class_filler(
            t,
            EquivalenceClasses::Atomic(vec![vec![0, 1, 2]]),
        )?),
        IOStatus::Party(_) => Ok(recursive_class_filler(
            t,
            EquivalenceClasses::Atomic(vec![vec![0], vec![1], vec![2]]),
        )?),
        IOStatus::Shared => Ok(get_input_class_helper_shared(t)?),
    }
}

fn recursive_class_filler(t: Type, sample_class: EquivalenceClasses) -> Result<EquivalenceClasses> {
    match t {
        Type::Scalar(_) | Type::Array(_, _) => Ok(sample_class),

        Type::Tuple(_) | Type::Vector(_, _) | Type::NamedTuple(_) => {
            let ts = get_types_vector(t)?;
            let mut current_class = vec![];
            for sub_t in ts {
                current_class.push(Arc::new(recursive_class_filler(
                    (*sub_t).clone(),
                    sample_class.clone(),
                )?));
            }
            Ok(EquivalenceClasses::Vector(current_class))
        }
    }
}

fn get_input_class_helper_shared(t: Type) -> Result<EquivalenceClasses> {
    match t {
        //The input should always be a tuple
        Type::Scalar(_) | Type::Array(_, _) | Type::Vector(_, _) | Type::NamedTuple(_) => {
            panic!("invalid input node");
        }

        //Check if the input is a tuple of size 3
        //Check all elements in the tuple have the same type
        Type::Tuple(_) => {
            let ts = get_types_vector(t)?;
            if ts.len() != 3 {
                panic!("invalid input node");
            }
            let mut current_class = vec![];
            let sample_class = vec![
                EquivalenceClasses::Atomic(vec![vec![1], vec![0, 2]]),
                EquivalenceClasses::Atomic(vec![vec![2], vec![0, 1]]),
                EquivalenceClasses::Atomic(vec![vec![0], vec![1, 2]]),
            ];
            for (index, sub_t) in ts.iter().enumerate() {
                if *sub_t != ts[0] {
                    panic!("invalid input node");
                }
                current_class.push(Arc::new(recursive_class_filler(
                    (**sub_t).clone(),
                    sample_class[index].clone(),
                )?));
            }
            Ok(EquivalenceClasses::Vector(current_class))
        }
    }
}

fn get_nop_class(
    previous_class: EquivalenceClasses,
    source_party: u64,
    destination_party: u64,
) -> Result<EquivalenceClasses> {
    match previous_class {
        EquivalenceClasses::Atomic(d) => {
            let mut current_class = d;
            for v in current_class.iter_mut() {
                if v.contains(&source_party) && !v.contains(&destination_party) {
                    v.push(destination_party);
                } else if v.contains(&destination_party) && !v.contains(&source_party) {
                    v.retain(|x| *x != destination_party);
                }
            }
            current_class.retain(|x| !x.is_empty());
            Ok(EquivalenceClasses::Atomic(current_class))
        }
        EquivalenceClasses::Vector(vd) => {
            let mut current_class = vec![];
            for e in vd {
                current_class.push(EquivalenceClassesPointer::new(get_nop_class(
                    (*e).clone(),
                    source_party,
                    destination_party,
                )?));
            }
            Ok(EquivalenceClasses::Vector(current_class))
        }
    }
}

//combine_class is used in the following nodes for now: Add, Subtract, Multiplication.
//Use the classes from two input nodes to generate the class for the result. The inputs must be Atomic.
//The input might be Public, Private, or Shared. Only the following possible cases will occur:
//0. Class1 op Class2 => If Class1 == Class2 => Class1 (or Class2), Otherwise:
//1. Either one is Public => Another one
//2. Either one is Private => Private i.e., [[0],[1],[2]]
//3. Both are Shared => Private (this is because two different shared classes will result in a Private result)
//Thus, the logic could be simplified: 1) if two inputs are equal, return either of them; 2) Otherwise, if anyone is public, return another one; 3) return Private;
fn combine_class(
    input1: EquivalenceClasses,
    input2: EquivalenceClasses,
) -> Result<EquivalenceClasses> {
    if input1 == input2 {
        return Ok(input1);
    }
    if input1 == EquivalenceClasses::Atomic(vec![vec![0, 1, 2]]) {
        return Ok(input2);
    }
    if input2 == EquivalenceClasses::Atomic(vec![vec![0, 1, 2]]) {
        Ok(input1)
    } else {
        Ok(EquivalenceClasses::Atomic(vec![vec![0], vec![1], vec![2]]))
    }
}

//This function checks protocol invariant.
//Only supports the NOP node up to now. If the sender and receiver already hold the same share, which means that this nop transmission should not be executed, this function will return false in this case.
#[allow(dead_code)]
pub(super) fn check_equivalence_class(
    context: Context,
    equivalence_classes: &HashMap<(u64, u64), EquivalenceClasses>,
    node: Node,
) -> Result<bool> {
    let dependencies = node.get_node_dependencies();
    let mut dependencies_class = vec![];
    for dependence_node in &dependencies {
        dependencies_class.push(
            equivalence_classes
                .get(&dependence_node.get_global_id())
                .expect("hashmap get error!")
                .clone(),
        );
    }
    let mut result = true;
    if node.get_operation() == Operation::NOP {
        let annotation = context.get_node_annotations(node)?;
        for single_communication in annotation {
            if let NodeAnnotation::Send(source_party, destination_party) = single_communication {
                result = check_equivalence_class_nop(
                    &dependencies_class[0],
                    source_party,
                    destination_party,
                )?;
                if !result {
                    break;
                }
            }
        }
    }
    Ok(result)
}

fn check_equivalence_class_nop(
    current_class: &EquivalenceClasses,
    source_party: u64,
    destination_party: u64,
) -> Result<bool> {
    match current_class {
        EquivalenceClasses::Atomic(d) => {
            let mut result = true;
            for v in d {
                if v.contains(&source_party) && v.contains(&destination_party) {
                    result = false;
                    break;
                }
            }
            Ok(result)
        }
        EquivalenceClasses::Vector(vd) => {
            let mut result = true;
            for e in vd {
                result = check_equivalence_class_nop(e, source_party, destination_party)?;
                if !result {
                    break;
                }
            }
            Ok(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_types::{array_type, scalar_type, tuple_type, vector_type, BIT, UINT64};
    use crate::data_values::Value;
    use crate::graphs::{create_context, create_unchecked_context, SliceElement};
    use crate::inline::inline_common::DepthOptimizationLevel;
    use crate::inline::inline_ops::{InlineConfig, InlineMode};
    use crate::mpc::mpc_compiler::{prepare_for_mpc_evaluation, IOStatus};
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn eq_equivalence_class_test() {
        let share0_12 = EquivalenceClasses::Atomic(vec![vec![0], vec![1, 2]]);
        let share0_21 = EquivalenceClasses::Atomic(vec![vec![0], vec![2, 1]]);
        let share12_0 = EquivalenceClasses::Atomic(vec![vec![0], vec![2, 1]]);
        let share1_02 = EquivalenceClasses::Atomic(vec![vec![1], vec![0, 2]]);

        let a = share0_12.clone();
        let b = share0_12.clone();
        assert_eq!(a, b);

        let a = share0_12.clone();
        let b = share0_21.clone();
        assert_eq!(a, b);

        let a = share0_12.clone();
        let b = share12_0.clone();
        assert_eq!(a, b);

        let a = EquivalenceClasses::Vector(vec![EquivalenceClassesPointer::new(share0_12.clone())]);
        let b = EquivalenceClasses::Vector(vec![EquivalenceClassesPointer::new(share0_12.clone())]);
        assert_eq!(a, b);

        let a = EquivalenceClasses::Vector(vec![EquivalenceClassesPointer::new(share0_12.clone())]);
        let b = share0_12.clone();
        assert!(a != b);

        let a = EquivalenceClasses::Vector(vec![
            EquivalenceClassesPointer::new(share0_12.clone()),
            EquivalenceClassesPointer::new(share0_12.clone()),
            EquivalenceClassesPointer::new(share0_12.clone()),
        ]);
        let b = a.clone();
        assert_eq!(a, b);

        let a = EquivalenceClasses::Vector(vec![
            EquivalenceClassesPointer::new(share0_12.clone()),
            EquivalenceClassesPointer::new(share0_12.clone()),
            EquivalenceClassesPointer::new(share0_12.clone()),
        ]);
        let b = EquivalenceClasses::Vector(vec![
            EquivalenceClassesPointer::new(share1_02.clone()),
            EquivalenceClassesPointer::new(share0_12.clone()),
            EquivalenceClassesPointer::new(share0_12.clone()),
        ]);
        assert!(a != b);
    }

    #[test]
    fn test_combine_class() {
        let public_class = EquivalenceClasses::Atomic(vec![vec![0, 1, 2]]);
        let private_class = EquivalenceClasses::Atomic(vec![vec![0], vec![1], vec![2]]);
        let share0_12 = EquivalenceClasses::Atomic(vec![vec![0], vec![1, 2]]);
        let share1_02 = EquivalenceClasses::Atomic(vec![vec![1], vec![0, 2]]);

        let a = private_class.clone();
        let b = private_class.clone();
        let ab = private_class.clone();
        assert_eq!(ab, combine_class(a, b).unwrap());

        let a = share0_12.clone();
        let b = share0_12.clone();
        let ab = share0_12.clone();
        assert_eq!(ab, combine_class(a, b).unwrap());

        let a = private_class.clone();
        let b = share0_12.clone();
        let ab = private_class.clone();
        assert_eq!(ab, combine_class(a, b).unwrap());

        let a = public_class.clone();
        let b = share0_12.clone();
        let ab = b.clone();
        assert_eq!(ab, combine_class(a, b).unwrap());

        let a = public_class.clone();
        let b = public_class.clone();
        let ab = public_class.clone();
        assert_eq!(ab, combine_class(a, b).unwrap());

        let a = share0_12.clone();
        let b = share1_02.clone();
        let ab = private_class.clone();
        assert_eq!(ab, combine_class(a, b).unwrap());
    }

    #[test]
    fn test_generate_equivalence_class() {
        let context1 = || -> Result<Context> {
            let context = create_unchecked_context()?;
            let g = context.create_graph()?;
            g.set_name("test_g1")?;
            let i1 = g.input(tuple_type(vec![
                scalar_type(BIT),
                scalar_type(BIT),
                scalar_type(BIT),
            ]))?;
            i1.set_name("i1")?;
            let i2 = g.input(tuple_type(vec![
                scalar_type(BIT),
                scalar_type(BIT),
                scalar_type(BIT),
            ]))?;
            i2.set_name("i2")?;
            let i3 = g.input(scalar_type(BIT))?;
            i3.set_name("i3")?;
            let i4 = g.input(vector_type(4, array_type(vec![1, 1, 1, 1], BIT)))?;
            i4.set_name("i4")?;
            let i5 = g.input(tuple_type(vec![
                scalar_type(BIT),
                scalar_type(BIT),
                scalar_type(BIT),
            ]))?;
            i5.set_name("i5")?;
            let add_op1 = g.tuple_get(i1.clone(), 0)?;
            add_op1.set_name("add_op1")?;
            let add_op2 = g.tuple_get(i2.clone(), 1)?;
            add_op2.set_name("add_op2")?;
            let add1 = g.add(add_op1.clone(), add_op2.clone())?;
            add1.set_name("add1")?;
            let subtract = g.subtract(add_op1.clone(), add_op2.clone())?;
            subtract.set_name("subtract")?;
            let multiply = g.multiply(add_op1.clone(), add_op2.clone())?;
            multiply.set_name("multiply")?;

            let rand1 = g.random(scalar_type(BIT))?;
            rand1.set_name("rand1")?;
            let rand2 = g.random(tuple_type(vec![
                scalar_type(BIT),
                scalar_type(BIT),
                array_type(vec![1, 1, 1, 1], BIT),
            ]))?;
            rand2.set_name("rand2")?;
            let nop_node = g.nop(rand1.clone())?;
            nop_node.set_name("nop_node")?;
            nop_node.add_annotation(NodeAnnotation::Send(0, 1))?;
            nop_node.add_annotation(NodeAnnotation::Send(0, 2))?;
            let prf1 = g.prf(
                nop_node,
                1234,
                vector_type(4, array_type(vec![1, 1, 1, 1], BIT)),
            )?;
            prf1.set_name("prf1")?;

            let tuple_get1 = g.tuple_get(i5.clone(), 1)?;
            tuple_get1.set_name("tuple_get1")?;
            let tuple_get2 = g.tuple_get(rand2.clone(), 0)?;
            tuple_get2.set_name("tuple_get2")?;
            let create_tuple = g.create_tuple(vec![tuple_get1, tuple_get2])?;
            create_tuple.set_name("create_tuple")?;
            Ok(context)
        }()
        .unwrap();

        let test_class1 = generate_equivalence_class(
            context1.clone(),
            vec![vec![
                IOStatus::Shared,
                IOStatus::Shared,
                IOStatus::Public,
                IOStatus::Party(0),
                IOStatus::Shared,
            ]],
        )
        .unwrap();
        let public_class = EquivalenceClasses::Atomic(vec![vec![0, 1, 2]]);
        let private_class = EquivalenceClasses::Atomic(vec![vec![0], vec![1], vec![2]]);
        let share0_12 = EquivalenceClasses::Atomic(vec![vec![0], vec![1, 2]]);
        let share1_02 = EquivalenceClasses::Atomic(vec![vec![1], vec![0, 2]]);
        let share2_01 = EquivalenceClasses::Atomic(vec![vec![2], vec![0, 1]]);

        let class_i1 = EquivalenceClasses::Vector(vec![
            Arc::new(share1_02.clone()),
            Arc::new(share2_01.clone()),
            Arc::new(share0_12.clone()),
        ]);
        let class_i2 = class_i1.clone();
        let class_i3 = public_class.clone();
        let class_i4 = EquivalenceClasses::Vector(vec![
            Arc::new(private_class.clone()),
            Arc::new(private_class.clone()),
            Arc::new(private_class.clone()),
            Arc::new(private_class.clone()),
        ]);
        let class_i5 = class_i1.clone();
        let class_add_op1 = share1_02.clone();
        let class_add_op2 = share2_01.clone();
        let class_add1 = private_class.clone();
        let class_subtract = private_class.clone();
        let class_multiply = private_class.clone();
        let class_rand1 = private_class.clone();
        let class_rand2 = EquivalenceClasses::Vector(vec![
            Arc::new(private_class.clone()),
            Arc::new(private_class.clone()),
            Arc::new(private_class.clone()),
        ]);
        let class_nop = class_i3.clone();
        let class_prf1 = EquivalenceClasses::Vector(vec![
            Arc::new(public_class.clone()),
            Arc::new(public_class.clone()),
            Arc::new(public_class.clone()),
            Arc::new(public_class.clone()),
        ]);

        let class_tuple_get1 = share2_01.clone();
        let class_tuple_get2 = private_class.clone();
        let class_create_tuple = EquivalenceClasses::Vector(vec![
            Arc::new(share2_01.clone()),
            Arc::new(private_class.clone()),
        ]);

        assert_eq!(
            *test_class1
                .get(
                    &context1
                        .retrieve_node(context1.retrieve_graph("test_g1").unwrap(), "i1")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class_i1
        );
        assert_eq!(
            *test_class1
                .get(
                    &context1
                        .retrieve_node(context1.retrieve_graph("test_g1").unwrap(), "i2")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class_i2
        );
        assert_eq!(
            *test_class1
                .get(
                    &context1
                        .retrieve_node(context1.retrieve_graph("test_g1").unwrap(), "i3")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class_i3
        );
        assert_eq!(
            *test_class1
                .get(
                    &context1
                        .retrieve_node(context1.retrieve_graph("test_g1").unwrap(), "i4")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class_i4
        );
        assert_eq!(
            *test_class1
                .get(
                    &context1
                        .retrieve_node(context1.retrieve_graph("test_g1").unwrap(), "i5")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class_i5
        );
        assert_eq!(
            *test_class1
                .get(
                    &context1
                        .retrieve_node(context1.retrieve_graph("test_g1").unwrap(), "add_op1")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class_add_op1
        );
        assert_eq!(
            *test_class1
                .get(
                    &context1
                        .retrieve_node(context1.retrieve_graph("test_g1").unwrap(), "add_op2")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class_add_op2
        );
        assert_eq!(
            *test_class1
                .get(
                    &context1
                        .retrieve_node(context1.retrieve_graph("test_g1").unwrap(), "add1")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class_add1
        );
        assert_eq!(
            *test_class1
                .get(
                    &context1
                        .retrieve_node(context1.retrieve_graph("test_g1").unwrap(), "subtract")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class_subtract
        );
        assert_eq!(
            *test_class1
                .get(
                    &context1
                        .retrieve_node(context1.retrieve_graph("test_g1").unwrap(), "multiply")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class_multiply
        );
        assert_eq!(
            *test_class1
                .get(
                    &context1
                        .retrieve_node(context1.retrieve_graph("test_g1").unwrap(), "rand1")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class_rand1
        );
        assert_eq!(
            *test_class1
                .get(
                    &context1
                        .retrieve_node(context1.retrieve_graph("test_g1").unwrap(), "rand2")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class_rand2
        );
        assert_eq!(
            *test_class1
                .get(
                    &context1
                        .retrieve_node(context1.retrieve_graph("test_g1").unwrap(), "nop_node")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class_nop
        );
        assert_eq!(
            *test_class1
                .get(
                    &context1
                        .retrieve_node(context1.retrieve_graph("test_g1").unwrap(), "prf1")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class_prf1
        );
        assert_eq!(
            *test_class1
                .get(
                    &context1
                        .retrieve_node(context1.retrieve_graph("test_g1").unwrap(), "tuple_get1")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class_tuple_get1
        );
        assert_eq!(
            *test_class1
                .get(
                    &context1
                        .retrieve_node(context1.retrieve_graph("test_g1").unwrap(), "tuple_get2")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class_tuple_get2
        );
        assert_eq!(
            *test_class1
                .get(
                    &context1
                        .retrieve_node(context1.retrieve_graph("test_g1").unwrap(), "create_tuple")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class_create_tuple
        );

        let context2 = || -> Result<Context> {
            let context = create_context()?;
            let g = context.create_graph()?;
            g.set_name("test_g2")?;
            let i1 = g.input(tuple_type(vec![
                scalar_type(UINT64),
                scalar_type(UINT64),
                scalar_type(UINT64),
            ]))?;
            i1.set_name("i1")?;
            let i2 = g.input(tuple_type(vec![
                scalar_type(UINT64),
                scalar_type(UINT64),
                scalar_type(UINT64),
            ]))?;
            i2.set_name("i2")?;
            let i3 = g.input(scalar_type(UINT64))?;
            i3.set_name("i3")?;

            let a2b = g.a2b(i3.clone())?;
            a2b.set_name("a2b")?;
            let b2a = g.b2a(a2b.clone(), UINT64)?;
            b2a.set_name("b2a")?;

            let tuple_get1 = g.tuple_get(i1.clone(), 0)?;
            tuple_get1.set_name("tuple_get1")?;
            let tuple_get2 = g.tuple_get(i2.clone(), 1)?;
            tuple_get2.set_name("tuple_get2")?;

            let repeat = g.repeat(tuple_get1.clone(), 4)?;
            repeat.set_name("repeat")?;

            let vector_to_array = g.vector_to_array(repeat.clone())?;
            vector_to_array.set_name("vector_to_array")?;

            let permuteaxe = g.permute_axes(vector_to_array.clone(), vec![0])?;
            permuteaxe.set_name("permuteaxe")?;

            let reshape = g.reshape(permuteaxe.clone(), array_type(vec![2, 2], UINT64))?;
            reshape.set_name("reshape")?;

            let stack = g.stack(vec![reshape.clone()], vec![1])?;
            stack.set_name("stack")?;

            let constant = g.constant(scalar_type(UINT64), Value::from_scalar(1, UINT64)?)?;
            constant.set_name("constant")?;

            let trunc = g.truncate(reshape.clone(), 2)?;
            trunc.set_name("trunc")?;

            let get_slice = g.get_slice(
                trunc.clone(),
                vec![
                    SliceElement::Ellipsis,
                    SliceElement::SubArray(None, None, Some(-1)),
                ],
            )?;
            get_slice.set_name("get_slice")?;

            let array_to_vector = g.array_to_vector(reshape.clone())?;
            array_to_vector.set_name("array_to_vector")?;

            let zip = g.zip(vec![array_to_vector.clone(), array_to_vector.clone()])?;
            zip.set_name("zip")?;

            let vector_get = g.vector_get(zip.clone(), i3.clone())?;
            vector_get.set_name("vector_get")?;

            let sum = g.sum(reshape.clone(), vec![0, 1])?;
            sum.set_name("sum")?;

            let matmul = g.matmul(reshape.clone(), reshape.clone())?;
            matmul.set_name("matmul")?;

            let get = g.get(vector_to_array.clone(), vec![2])?;
            get.set_name("get")?;

            let dot = g.dot(vector_to_array.clone(), vector_to_array.clone())?;
            dot.set_name("dot")?;

            let create_named_tuple = g.create_named_tuple(vec![
                ("dot".to_string(), dot.clone()),
                ("get".to_string(), get.clone()),
            ])?;
            create_named_tuple.set_name("create_named_tuple")?;

            let named_tuple_get =
                g.named_tuple_get(create_named_tuple.clone(), "get".to_string())?;
            named_tuple_get.set_name("named_tuple_get")?;

            let create_vector =
                g.create_vector(scalar_type(UINT64), vec![i3.clone(), i3.clone()])?;
            create_vector.set_name("create_vector")?;
            Ok(context)
        }()
        .unwrap();

        let test_class2 = generate_equivalence_class(
            context2.clone(),
            vec![vec![IOStatus::Shared, IOStatus::Shared, IOStatus::Public]],
        )
        .unwrap();

        let class2_i1 = EquivalenceClasses::Vector(vec![
            Arc::new(share1_02.clone()),
            Arc::new(share2_01.clone()),
            Arc::new(share0_12.clone()),
        ]);
        let class2_i2 = class_i1.clone();
        let class2_i3 = public_class.clone();
        let class2_a2b = public_class.clone();
        let class2_b2a = public_class.clone();
        let class2_tuple_get1 = share1_02.clone();
        let class2_tuple_get2 = share2_01.clone();
        let class2_repeat = EquivalenceClasses::Vector(vec![
            Arc::new(share1_02.clone()),
            Arc::new(share1_02.clone()),
            Arc::new(share1_02.clone()),
            Arc::new(share1_02.clone()),
        ]);
        let class2_vector_to_array = share1_02.clone();
        let class2_permuteaxe = share1_02.clone();
        let class2_reshape = share1_02.clone();
        let class2_stack = share1_02.clone();
        let class2_constant = public_class.clone();
        let class2_trunc = share1_02.clone();
        let class2_get_slice = share1_02.clone();
        let class2_array_to_vector = EquivalenceClasses::Vector(vec![
            Arc::new(share1_02.clone()),
            Arc::new(share1_02.clone()),
        ]);
        let class2_zip = EquivalenceClasses::Vector(vec![
            Arc::new(class2_array_to_vector.clone()),
            Arc::new(class2_array_to_vector.clone()),
        ]);
        let class2_vector_get = EquivalenceClasses::Vector(vec![
            Arc::new(share1_02.clone()),
            Arc::new(share1_02.clone()),
        ]);
        let class2_sum = share1_02.clone();
        let class2_matmul = share1_02.clone();
        let class2_get = share1_02.clone();
        let class2_dot = share1_02.clone();
        let class2_create_named_tuple = EquivalenceClasses::Vector(vec![
            Arc::new(share1_02.clone()),
            Arc::new(share1_02.clone()),
        ]);
        let class2_named_tuple_get = share1_02.clone();
        let class2_create_vector = EquivalenceClasses::Vector(vec![
            Arc::new(public_class.clone()),
            Arc::new(public_class.clone()),
        ]);
        assert_eq!(
            *test_class2
                .get(
                    &context2
                        .retrieve_node(context2.retrieve_graph("test_g2").unwrap(), "i1")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class2_i1
        );
        assert_eq!(
            *test_class2
                .get(
                    &context2
                        .retrieve_node(context2.retrieve_graph("test_g2").unwrap(), "i2")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class2_i2
        );
        assert_eq!(
            *test_class2
                .get(
                    &context2
                        .retrieve_node(context2.retrieve_graph("test_g2").unwrap(), "i3")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class2_i3
        );
        assert_eq!(
            *test_class2
                .get(
                    &context2
                        .retrieve_node(context2.retrieve_graph("test_g2").unwrap(), "a2b")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class2_a2b
        );
        assert_eq!(
            *test_class2
                .get(
                    &context2
                        .retrieve_node(context2.retrieve_graph("test_g2").unwrap(), "b2a")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class2_b2a
        );
        assert_eq!(
            *test_class2
                .get(
                    &context2
                        .retrieve_node(context2.retrieve_graph("test_g2").unwrap(), "tuple_get1")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class2_tuple_get1
        );
        assert_eq!(
            *test_class2
                .get(
                    &context2
                        .retrieve_node(context2.retrieve_graph("test_g2").unwrap(), "tuple_get2")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class2_tuple_get2
        );
        assert_eq!(
            *test_class2
                .get(
                    &context2
                        .retrieve_node(context2.retrieve_graph("test_g2").unwrap(), "repeat")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class2_repeat
        );
        assert_eq!(
            *test_class2
                .get(
                    &context2
                        .retrieve_node(
                            context2.retrieve_graph("test_g2").unwrap(),
                            "vector_to_array"
                        )
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class2_vector_to_array
        );
        assert_eq!(
            *test_class2
                .get(
                    &context2
                        .retrieve_node(context2.retrieve_graph("test_g2").unwrap(), "permuteaxe")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class2_permuteaxe
        );
        assert_eq!(
            *test_class2
                .get(
                    &context2
                        .retrieve_node(context2.retrieve_graph("test_g2").unwrap(), "reshape")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class2_reshape
        );
        assert_eq!(
            *test_class2
                .get(
                    &context2
                        .retrieve_node(context2.retrieve_graph("test_g2").unwrap(), "stack")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class2_stack
        );
        assert_eq!(
            *test_class2
                .get(
                    &context2
                        .retrieve_node(context2.retrieve_graph("test_g2").unwrap(), "constant")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class2_constant
        );
        assert_eq!(
            *test_class2
                .get(
                    &context2
                        .retrieve_node(context2.retrieve_graph("test_g2").unwrap(), "trunc")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class2_trunc
        );
        assert_eq!(
            *test_class2
                .get(
                    &context2
                        .retrieve_node(context2.retrieve_graph("test_g2").unwrap(), "get_slice")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class2_get_slice
        );
        assert_eq!(
            *test_class2
                .get(
                    &context2
                        .retrieve_node(
                            context2.retrieve_graph("test_g2").unwrap(),
                            "array_to_vector"
                        )
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class2_array_to_vector
        );
        assert_eq!(
            *test_class2
                .get(
                    &context2
                        .retrieve_node(context2.retrieve_graph("test_g2").unwrap(), "zip")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class2_zip
        );
        assert_eq!(
            *test_class2
                .get(
                    &context2
                        .retrieve_node(context2.retrieve_graph("test_g2").unwrap(), "vector_get")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class2_vector_get
        );
        assert_eq!(
            *test_class2
                .get(
                    &context2
                        .retrieve_node(context2.retrieve_graph("test_g2").unwrap(), "sum")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class2_sum
        );
        assert_eq!(
            *test_class2
                .get(
                    &context2
                        .retrieve_node(context2.retrieve_graph("test_g2").unwrap(), "matmul")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class2_matmul
        );
        assert_eq!(
            *test_class2
                .get(
                    &context2
                        .retrieve_node(context2.retrieve_graph("test_g2").unwrap(), "get")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class2_get
        );
        assert_eq!(
            *test_class2
                .get(
                    &context2
                        .retrieve_node(context2.retrieve_graph("test_g2").unwrap(), "dot")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class2_dot
        );
        assert_eq!(
            *test_class2
                .get(
                    &context2
                        .retrieve_node(
                            context2.retrieve_graph("test_g2").unwrap(),
                            "create_named_tuple"
                        )
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class2_create_named_tuple
        );
        assert_eq!(
            *test_class2
                .get(
                    &context2
                        .retrieve_node(
                            context2.retrieve_graph("test_g2").unwrap(),
                            "named_tuple_get"
                        )
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class2_named_tuple_get
        );
        assert_eq!(
            *test_class2
                .get(
                    &context2
                        .retrieve_node(context2.retrieve_graph("test_g2").unwrap(), "create_vector")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class2_create_vector
        );
    }

    #[test]
    fn test_class_compiled_graph() {
        let c = create_context().unwrap();
        let g = c.create_graph().unwrap();
        g.input(scalar_type(UINT64))
            .unwrap()
            .multiply(g.input(scalar_type(UINT64)).unwrap())
            .unwrap()
            .set_as_output()
            .unwrap();
        g.finalize().unwrap();
        c.set_main_graph(g).unwrap();
        c.finalize().unwrap();
        {
            let compiled_c = prepare_for_mpc_evaluation(
                c.clone(),
                vec![vec![IOStatus::Shared, IOStatus::Public]],
                vec![vec![]],
                InlineConfig {
                    default_mode: InlineMode::DepthOptimized(DepthOptimizationLevel::Default),
                    ..Default::default()
                },
            )
            .unwrap();
            let test_class1 = generate_equivalence_class(
                compiled_c.clone(),
                vec![vec![IOStatus::Shared, IOStatus::Public]],
            )
            .unwrap();

            let public_class = EquivalenceClasses::Atomic(vec![vec![0, 1, 2]]);
            let private_class = EquivalenceClasses::Atomic(vec![vec![0], vec![1], vec![2]]);
            let share0_12 = EquivalenceClasses::Atomic(vec![vec![0], vec![1, 2]]);
            let share1_02 = EquivalenceClasses::Atomic(vec![vec![1], vec![0, 2]]);
            let share2_01 = EquivalenceClasses::Atomic(vec![vec![2], vec![0, 1]]);
            let shared = EquivalenceClasses::Vector(vec![
                Arc::new(share1_02.clone()),
                Arc::new(share2_01.clone()),
                Arc::new(share0_12.clone()),
            ]);
            assert_eq!(*test_class1.get(&(0, 0)).unwrap(), private_class.clone());
            assert_eq!(*test_class1.get(&(0, 1)).unwrap(), share1_02.clone());
            assert_eq!(*test_class1.get(&(0, 2)).unwrap(), private_class.clone());
            assert_eq!(*test_class1.get(&(0, 3)).unwrap(), share2_01.clone());
            assert_eq!(*test_class1.get(&(0, 4)).unwrap(), private_class.clone());
            assert_eq!(*test_class1.get(&(0, 5)).unwrap(), share0_12.clone());
            assert_eq!(*test_class1.get(&(0, 6)).unwrap(), shared.clone());
            assert_eq!(*test_class1.get(&(0, 7)).unwrap(), shared.clone());
            assert_eq!(*test_class1.get(&(0, 8)).unwrap(), public_class.clone());
            assert_eq!(*test_class1.get(&(0, 9)).unwrap(), share1_02.clone());
            assert_eq!(*test_class1.get(&(0, 10)).unwrap(), share1_02.clone());
            assert_eq!(*test_class1.get(&(0, 11)).unwrap(), share2_01.clone());
            assert_eq!(*test_class1.get(&(0, 12)).unwrap(), share2_01.clone());
            assert_eq!(*test_class1.get(&(0, 13)).unwrap(), share0_12.clone());
            assert_eq!(*test_class1.get(&(0, 14)).unwrap(), share0_12.clone());
            assert_eq!(*test_class1.get(&(0, 15)).unwrap(), shared.clone());
        }
        {
            let compiled_c = prepare_for_mpc_evaluation(
                c.clone(),
                vec![vec![IOStatus::Shared, IOStatus::Public]],
                vec![vec![IOStatus::Party(0)]],
                InlineConfig {
                    default_mode: InlineMode::DepthOptimized(DepthOptimizationLevel::Default),
                    ..Default::default()
                },
            )
            .unwrap();
            let test_class1 = generate_equivalence_class(
                compiled_c.clone(),
                vec![vec![IOStatus::Shared, IOStatus::Public]],
            )
            .unwrap();

            let public_class = EquivalenceClasses::Atomic(vec![vec![0, 1, 2]]);
            let private_class = EquivalenceClasses::Atomic(vec![vec![0], vec![1], vec![2]]);
            let share0_12 = EquivalenceClasses::Atomic(vec![vec![0], vec![1, 2]]);
            let share1_02 = EquivalenceClasses::Atomic(vec![vec![1], vec![0, 2]]);
            let share2_01 = EquivalenceClasses::Atomic(vec![vec![2], vec![0, 1]]);
            let shared = EquivalenceClasses::Vector(vec![
                Arc::new(share1_02.clone()),
                Arc::new(share2_01.clone()),
                Arc::new(share0_12.clone()),
            ]);
            // PRF keys
            assert_eq!(*test_class1.get(&(0, 0)).unwrap(), private_class.clone());
            assert_eq!(*test_class1.get(&(0, 1)).unwrap(), share1_02.clone());
            assert_eq!(*test_class1.get(&(0, 2)).unwrap(), private_class.clone());
            assert_eq!(*test_class1.get(&(0, 3)).unwrap(), share2_01.clone());
            assert_eq!(*test_class1.get(&(0, 4)).unwrap(), private_class.clone());
            assert_eq!(*test_class1.get(&(0, 5)).unwrap(), share0_12.clone());
            // Create PRF triple
            assert_eq!(*test_class1.get(&(0, 6)).unwrap(), shared.clone());
            // Shared input
            assert_eq!(*test_class1.get(&(0, 7)).unwrap(), shared.clone());
            // Public input
            assert_eq!(*test_class1.get(&(0, 8)).unwrap(), public_class.clone());
            // Extract share 0
            assert_eq!(*test_class1.get(&(0, 9)).unwrap(), share1_02.clone());
            // Multiply share 0 by the public value
            assert_eq!(*test_class1.get(&(0, 10)).unwrap(), share1_02.clone());
            // Extract share 1
            assert_eq!(*test_class1.get(&(0, 11)).unwrap(), share2_01.clone());
            // Multiply share 1 by the public value
            assert_eq!(*test_class1.get(&(0, 12)).unwrap(), share2_01.clone());
            // Extract share 2
            assert_eq!(*test_class1.get(&(0, 13)).unwrap(), share0_12.clone());
            // Multiply share 2 by the public value
            assert_eq!(*test_class1.get(&(0, 14)).unwrap(), share0_12.clone());
            // Shared product
            assert_eq!(*test_class1.get(&(0, 15)).unwrap(), shared.clone());
            // Extract shares
            assert_eq!(*test_class1.get(&(0, 16)).unwrap(), share1_02.clone());
            assert_eq!(*test_class1.get(&(0, 17)).unwrap(), share2_01.clone());
            assert_eq!(*test_class1.get(&(0, 18)).unwrap(), share0_12.clone());
            // Revealing
            // Share 2 is sent to party 0, thus becoming public
            assert_eq!(*test_class1.get(&(0, 19)).unwrap(), public_class.clone());
            // Sum of shares 0 and 1 must be private (party 0 has the correct sum)
            assert_eq!(*test_class1.get(&(0, 20)).unwrap(), private_class.clone());
            // Sum of shares 0, 1 and 2 must be private (party 0 has the correct sum)
            assert_eq!(*test_class1.get(&(0, 21)).unwrap(), private_class.clone());
            // There should be no other nodes
            assert!(test_class1.get(&(0, 22)).is_none());
        }
        {
            let compiled_c = prepare_for_mpc_evaluation(
                c.clone(),
                vec![vec![IOStatus::Shared, IOStatus::Public]],
                vec![vec![IOStatus::Party(0), IOStatus::Party(2)]],
                InlineConfig {
                    default_mode: InlineMode::DepthOptimized(DepthOptimizationLevel::Default),
                    ..Default::default()
                },
            )
            .unwrap();
            let test_class1 = generate_equivalence_class(
                compiled_c.clone(),
                vec![vec![IOStatus::Shared, IOStatus::Public]],
            )
            .unwrap();

            let public_class = EquivalenceClasses::Atomic(vec![vec![0, 1, 2]]);
            let private_class = EquivalenceClasses::Atomic(vec![vec![0], vec![1], vec![2]]);
            let share0_12 = EquivalenceClasses::Atomic(vec![vec![0], vec![1, 2]]);
            let share1_02 = EquivalenceClasses::Atomic(vec![vec![1], vec![0, 2]]);
            let share2_01 = EquivalenceClasses::Atomic(vec![vec![2], vec![0, 1]]);
            let shared = EquivalenceClasses::Vector(vec![
                Arc::new(share1_02.clone()),
                Arc::new(share2_01.clone()),
                Arc::new(share0_12.clone()),
            ]);
            // PRF keys
            assert_eq!(*test_class1.get(&(0, 0)).unwrap(), private_class.clone());
            assert_eq!(*test_class1.get(&(0, 1)).unwrap(), share1_02.clone());
            assert_eq!(*test_class1.get(&(0, 2)).unwrap(), private_class.clone());
            assert_eq!(*test_class1.get(&(0, 3)).unwrap(), share2_01.clone());
            assert_eq!(*test_class1.get(&(0, 4)).unwrap(), private_class.clone());
            assert_eq!(*test_class1.get(&(0, 5)).unwrap(), share0_12.clone());
            // Create PRF triple
            assert_eq!(*test_class1.get(&(0, 6)).unwrap(), shared.clone());
            // Shared input
            assert_eq!(*test_class1.get(&(0, 7)).unwrap(), shared.clone());
            // Public input
            assert_eq!(*test_class1.get(&(0, 8)).unwrap(), public_class.clone());
            // Extract share 0
            assert_eq!(*test_class1.get(&(0, 9)).unwrap(), share1_02.clone());
            // Multiply share 0 by the public value
            assert_eq!(*test_class1.get(&(0, 10)).unwrap(), share1_02.clone());
            // Extract share 1
            assert_eq!(*test_class1.get(&(0, 11)).unwrap(), share2_01.clone());
            // Multiply share 1 by the public value
            assert_eq!(*test_class1.get(&(0, 12)).unwrap(), share2_01.clone());
            // Extract share 2
            assert_eq!(*test_class1.get(&(0, 13)).unwrap(), share0_12.clone());
            // Multiply share 2 by the public value
            assert_eq!(*test_class1.get(&(0, 14)).unwrap(), share0_12.clone());
            // Shared product
            assert_eq!(*test_class1.get(&(0, 15)).unwrap(), shared.clone());
            // Extract shares
            assert_eq!(*test_class1.get(&(0, 16)).unwrap(), share1_02.clone());
            assert_eq!(*test_class1.get(&(0, 17)).unwrap(), share2_01.clone());
            assert_eq!(*test_class1.get(&(0, 18)).unwrap(), share0_12.clone());
            // Revealing
            // Share 2 is sent to party 0, thus becoming public
            assert_eq!(*test_class1.get(&(0, 19)).unwrap(), public_class.clone());
            // Sum of shares 0 and 1 must be private (party 0 has the correct sum)
            assert_eq!(*test_class1.get(&(0, 20)).unwrap(), private_class.clone());
            // Sum of shares 0, 1 and 2 must be private (party 0 has the correct sum)
            assert_eq!(*test_class1.get(&(0, 21)).unwrap(), private_class.clone());
            // Send the revealed value to another party
            assert_eq!(*test_class1.get(&(0, 22)).unwrap(), share1_02.clone());
            // Output node can't have Send annotation
            assert_eq!(*test_class1.get(&(0, 23)).unwrap(), share1_02.clone());
            // There should be no other nodes
            assert!(test_class1.get(&(0, 24)).is_none());
        }
    }

    #[test]
    fn test_check_equivalence_class() {
        //following will only test nop node
        let context1 = || -> Result<Context> {
            let context = create_unchecked_context()?;
            let g = context.create_graph()?;
            let i1 = g.input(tuple_type(vec![
                scalar_type(BIT),
                scalar_type(BIT),
                scalar_type(BIT),
            ]))?;
            let nop_node = g.nop(i1.clone())?;
            nop_node.add_annotation(NodeAnnotation::Send(0, 1))?;
            Ok(context)
        }()
        .unwrap();
        let test_class1 = generate_equivalence_class(
            context1.clone(),
            vec![vec![IOStatus::Shared, IOStatus::Shared]],
        )
        .unwrap();

        let context2 = || -> Result<Context> {
            let context = create_unchecked_context()?;
            let g = context.create_graph()?;
            let i1 = g.random(scalar_type(BIT))?;
            let nop_node = g.nop(i1.clone())?;
            nop_node.add_annotation(NodeAnnotation::Send(0, 1))?;
            Ok(context)
        }()
        .unwrap();
        let test_class2 = generate_equivalence_class(
            context2.clone(),
            vec![vec![IOStatus::Shared, IOStatus::Shared]],
        )
        .unwrap();

        assert_eq!(
            helper_equivalence_class(context1.clone(), &test_class1).unwrap(),
            false
        );
        assert_eq!(
            helper_equivalence_class(context2.clone(), &test_class2).unwrap(),
            true
        );
    }

    fn helper_equivalence_class(
        context: Context,
        equivalence_classes: &HashMap<(u64, u64), EquivalenceClasses>,
    ) -> Result<bool> {
        let mut result = true;
        let graphs = context.get_graphs();
        for graph in graphs {
            let nodes = graph.get_nodes();
            for node in nodes {
                result = check_equivalence_class(context.clone(), equivalence_classes, node)?;
                if !result {
                    break;
                }
            }
        }
        Ok(result)
    }
}
