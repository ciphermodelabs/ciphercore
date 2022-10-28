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

impl EquivalenceClasses {
    fn is_atomic(&self) -> bool {
        matches!(self, EquivalenceClasses::Atomic(_))
    }

    fn is_vector(&self) -> bool {
        matches!(self, EquivalenceClasses::Vector(_))
    }

    fn get_class_vector(&self) -> Vec<Arc<EquivalenceClasses>> {
        if let EquivalenceClasses::Vector(v) = self {
            (*v).clone()
        } else {
            panic!("This class is not a vector");
        }
    }
}

pub(super) fn public_class() -> EquivalenceClasses {
    EquivalenceClasses::Atomic(vec![vec![0, 1, 2]])
}

pub(super) fn private_class() -> EquivalenceClasses {
    EquivalenceClasses::Atomic(vec![vec![0], vec![1], vec![2]])
}

pub(super) fn share0_class() -> EquivalenceClasses {
    EquivalenceClasses::Atomic(vec![vec![1], vec![0, 2]])
}

pub(super) fn share1_class() -> EquivalenceClasses {
    EquivalenceClasses::Atomic(vec![vec![2], vec![0, 1]])
}

pub(super) fn share2_class() -> EquivalenceClasses {
    EquivalenceClasses::Atomic(vec![vec![0], vec![1, 2]])
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

#[allow(dead_code)]
pub(super) fn vector_class(v: Vec<EquivalenceClasses>) -> EquivalenceClasses {
    let mut dependencies = vec![];
    for class in v {
        dependencies.push(Arc::new(class));
    }
    EquivalenceClasses::Vector(dependencies)
}

fn flatten_classes(input_class: EquivalenceClasses) -> Vec<EquivalenceClasses> {
    match input_class {
        EquivalenceClasses::Atomic(_) => vec![input_class],
        EquivalenceClasses::Vector(v) => {
            let mut result_vec = vec![];
            for class in v {
                result_vec.extend(flatten_classes((*class).clone()));
            }
            result_vec
        }
    }
}

fn unflatten_classes(
    flattened_classes: &[EquivalenceClasses],
    t: Type,
    position: &mut u64,
) -> EquivalenceClasses {
    match t {
        Type::Array(_, _) | Type::Scalar(_) => {
            *position += 1;
            flattened_classes[(*position - 1) as usize].clone()
        }
        Type::NamedTuple(v) => {
            let mut class_vec = vec![];
            for (_, sub_t) in v {
                class_vec.push(unflatten_classes(
                    flattened_classes,
                    (*sub_t).clone(),
                    position,
                ));
            }
            vector_class(class_vec)
        }
        Type::Tuple(v) => {
            let mut class_vec = vec![];
            for sub_t in v {
                class_vec.push(unflatten_classes(
                    flattened_classes,
                    (*sub_t).clone(),
                    position,
                ));
            }
            vector_class(class_vec)
        }
        Type::Vector(n, sub_t) => {
            let mut class_vec = vec![];
            for _ in 0..n {
                class_vec.push(unflatten_classes(
                    flattened_classes,
                    (*sub_t).clone(),
                    position,
                ));
            }
            vector_class(class_vec)
        }
    }
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
                let node_id = dependence_node.get_global_id();
                let op = dependence_node.get_operation();
                dependencies_class.push(
                    equivalence_classes
                        .get(&node_id)
                        .unwrap_or_else(|| {
                            panic!(
                                "{} node {:?} wasn't added to equivalence classes",
                                op, node_id
                            )
                        })
                        .clone(),
                );
            }
            let result_class = match node.get_operation() {
                Operation::Input(input_type) => {
                    let result_class =
                        get_input_class(input_type, &input_party_map[0][input_count])?;
                    input_count += 1;
                    result_class
                }
                Operation::CreateTuple
                | Operation::CreateNamedTuple(_)
                | Operation::CreateVector(_) => vector_class(dependencies_class),

                Operation::TupleGet(field_id) => {
                    let input_class = dependencies_class[0].clone();
                    if !input_class.is_vector() {
                        panic!("TupleGet input class should be Vector")
                    }
                    (*input_class.get_class_vector()[field_id as usize]).clone()
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
                    let input_class = dependencies_class[0].clone();
                    if !input_class.is_vector() {
                        panic!("NamedTupleGet input class should be Vector")
                    }
                    (*input_class.get_class_vector()[field_id_raw as usize]).clone()
                }

                Operation::Random(t) => recursive_class_filler(
                    t,
                    EquivalenceClasses::Atomic(vec![vec![0], vec![1], vec![2]]),
                )?,

                Operation::NOP => {
                    let mut previous_class = dependencies_class[0].clone();
                    let annotation = context.get_node_annotations(node.clone())?;
                    for single_communication in annotation {
                        if let NodeAnnotation::Send(source_party, destination_party) =
                            single_communication
                        {
                            previous_class =
                                get_nop_class(previous_class, source_party, destination_party)?;
                        }
                    }
                    previous_class
                }

                Operation::PRF(_, t) => recursive_class_filler(t, dependencies_class[0].clone())?,

                Operation::Add
                | Operation::Subtract
                | Operation::Multiply
                | Operation::MixedMultiply
                | Operation::Dot
                | Operation::Matmul
                | Operation::CuckooHash
                | Operation::Gather(_) => {
                    if !dependencies_class[0].is_atomic() {
                        panic!(
                            "{} first input class should be Atomic",
                            node.get_operation()
                        )
                    }
                    if !dependencies_class[1].is_atomic() {
                        panic!(
                            "{} second input class should be Atomic",
                            node.get_operation()
                        )
                    }
                    combine_class(dependencies_class[0].clone(), dependencies_class[1].clone())?
                }

                Operation::Truncate(_)
                | Operation::Sum(_)
                | Operation::Get(_)
                | Operation::GetSlice(_)
                | Operation::A2B
                | Operation::B2A(_)
                | Operation::InversePermutation
                | Operation::PermuteAxes(_) => {
                    if !dependencies_class[0].is_atomic() {
                        panic!("{} input class should be Atomic", node.get_operation())
                    }
                    dependencies_class[0].clone()
                }

                Operation::Reshape(result_type) => {
                    let input_classes = flatten_classes(dependencies_class[0].clone());
                    unflatten_classes(&input_classes, result_type, &mut 0)
                }

                Operation::Stack(_) => {
                    let mut result_class = dependencies_class[0].clone();
                    if !result_class.is_atomic() {
                        panic!("Stack input classes must be Atomic");
                    }
                    for class in dependencies_class.iter().skip(1) {
                        if !class.is_atomic() {
                            panic!("Stack input classes must be Atomic");
                        }
                        result_class = combine_class(result_class, (*class).clone())?;
                    }
                    result_class
                }

                Operation::VectorToArray => {
                    let input_class = dependencies_class[0].clone();
                    if !input_class.is_vector() {
                        panic!("VectorToArray input class must be Vector");
                    }
                    let class_vec = input_class.get_class_vector();
                    let mut result_class = (*class_vec[0]).clone();
                    for e in class_vec.iter().skip(1) {
                        result_class = combine_class(result_class, (**e).clone())?;
                    }
                    result_class
                }

                Operation::Zip => {
                    let mut result_classes = vec![];
                    let mut index = 0;
                    'result_entries: loop {
                        let mut row = vec![];
                        for dependency_class in dependencies_class.clone() {
                            if !dependency_class.is_vector() {
                                panic!("Zip input class must be Vector");
                            }
                            let v = dependency_class.get_class_vector();
                            if v.len() <= index {
                                break 'result_entries;
                            }
                            row.push(v[index].clone());
                        }
                        result_classes.push(EquivalenceClasses::Vector(row));
                        index += 1;
                    }

                    vector_class(result_classes)
                }

                Operation::Constant(t, _) => recursive_class_filler(t, public_class())?,

                Operation::Repeat(n) => {
                    let mut result_classes = vec![];
                    for _ in 0..n {
                        result_classes.push(dependencies_class[0].clone());
                    }
                    vector_class(result_classes)
                }

                Operation::ArrayToVector => {
                    let input_class = dependencies_class[0].clone();
                    if !input_class.is_atomic() {
                        panic!("ArrayToVector input class should be Atomic");
                    }
                    let mut classes = vec![];
                    let dependency_node = dependencies[0].clone();
                    let shape = dependency_node.get_type()?.get_shape();
                    for _ in 0..shape[0] {
                        classes.push(input_class.clone());
                    }
                    vector_class(classes)
                }

                Operation::VectorGet => {
                    let input_class = dependencies_class[0].clone();
                    if !input_class.is_vector() {
                        panic!("VectorGet input class should be Vector");
                    }
                    let v = input_class.get_class_vector();
                    let result_class = (*v[0]).clone();
                    for class in v {
                        if result_class != *class {
                            panic!("VectorGet input class contains different EquivalenceClasses");
                        }
                    }
                    result_class
                }
                Operation::RandomPermutation(_) | Operation::CuckooToPermutation => private_class(),
                Operation::DecomposeSwitchingMap(_) => vector_class(vec![
                    private_class(),
                    vector_class(vec![private_class(), private_class()]),
                    private_class(),
                ]),
                Operation::SegmentCumSum => combine_class(
                    combine_class(dependencies_class[0].clone(), dependencies_class[1].clone())?,
                    dependencies_class[2].clone(),
                )?,
                _ => return Err(runtime_error!("Operation is not supported")),
            };
            equivalence_classes.insert(node.get_global_id(), result_class);
        }
    }
    Ok(equivalence_classes)
}

fn get_input_class(t: Type, input_party: &IOStatus) -> Result<EquivalenceClasses> {
    match input_party {
        IOStatus::Public => Ok(recursive_class_filler(t, public_class())?),
        IOStatus::Party(_) => Ok(recursive_class_filler(t, private_class())?),
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
            let sample_class = vec![share0_class(), share1_class(), share2_class()];
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
    if !input1.is_atomic() || !input2.is_atomic() {
        panic!("Only Atomic classes can be combined");
    }
    if input1 == input2 {
        return Ok(input1);
    }
    if input1 == public_class() {
        return Ok(input2);
    }
    if input2 == public_class() {
        return Ok(input1);
    }
    Ok(private_class())
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
    use crate::graphs::{create_context, create_unchecked_context, Graph, SliceElement};
    use crate::inline::inline_common::DepthOptimizationLevel;
    use crate::inline::inline_ops::{InlineConfig, InlineMode};
    use crate::mpc::mpc_compiler::{prepare_for_mpc_evaluation, IOStatus};
    use std::collections::HashMap;

    type ClassesMap = HashMap<(u64, u64), EquivalenceClasses>;

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

        let a = vector_class(vec![share0_12.clone()]);
        let b = vector_class(vec![share0_12.clone()]);
        assert_eq!(a, b);

        let a = vector_class(vec![share0_12.clone()]);
        let b = share0_12.clone();
        assert!(a != b);

        let a = vector_class(vec![share0_12.clone(); 3]);
        let b = a.clone();
        assert_eq!(a, b);

        let a = vector_class(vec![share0_12.clone(); 3]);
        let b = vector_class(vec![
            share1_02.clone(),
            share0_12.clone(),
            share0_12.clone(),
        ]);
        assert!(a != b);
    }

    #[test]
    fn test_combine_class() {
        let a = private_class();
        let b = private_class();
        let ab = private_class();
        assert_eq!(ab, combine_class(a, b).unwrap());

        let a = share2_class();
        let b = share2_class();
        let ab = share2_class();
        assert_eq!(ab, combine_class(a, b).unwrap());

        let a = private_class();
        let b = share2_class();
        let ab = private_class();
        assert_eq!(ab, combine_class(a, b).unwrap());

        let a = public_class();
        let b = share2_class();
        let ab = b.clone();
        assert_eq!(ab, combine_class(a, b).unwrap());

        let a = public_class();
        let b = public_class();
        let ab = public_class();
        assert_eq!(ab, combine_class(a, b).unwrap());

        let a = share2_class();
        let b = share0_class();
        let ab = private_class();
        assert_eq!(ab, combine_class(a, b).unwrap());
    }

    fn get_class_from_name(
        graph: &Graph,
        classes: &ClassesMap,
        name: &str,
    ) -> Result<EquivalenceClasses> {
        let node = graph.retrieve_node(name)?;
        Ok((*classes.get(&node.get_global_id()).unwrap()).clone())
    }

    #[test]
    fn test_input() {
        || -> Result<()> {
            let c = create_unchecked_context()?;
            let g = c.create_graph()?;
            let t = array_type(vec![10], BIT);
            g.input(tuple_type(vec![t.clone(), t.clone(), t.clone()]))?
                .set_name("i1")?;
            g.input(t.clone())?.set_name("i2")?;
            g.input(t.clone())?.set_name("i3")?;

            let result_classes = generate_equivalence_class(
                c.clone(),
                vec![vec![IOStatus::Shared, IOStatus::Public, IOStatus::Party(0)]],
            )?;

            assert_eq!(
                get_class_from_name(&g, &result_classes, "i1")?,
                vector_class(vec![share0_class(), share1_class(), share2_class()])
            );
            assert_eq!(
                get_class_from_name(&g, &result_classes, "i2")?,
                public_class()
            );
            assert_eq!(
                get_class_from_name(&g, &result_classes, "i3")?,
                private_class()
            );

            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_stack() {
        || -> Result<()> {
            let c = create_unchecked_context()?;
            let g = c.create_graph()?;
            let t = array_type(vec![10], BIT);
            let shared_tuple = g.input(tuple_type(vec![t.clone(), t.clone(), t.clone()]))?;
            let public_array = g.input(t.clone())?;
            let private_array = g.input(t.clone())?;

            g.stack(vec![public_array.clone(), private_array.clone()], vec![2])?
                .set_name("stack1")?;
            g.stack(vec![public_array.clone(), public_array.clone()], vec![2])?
                .set_name("stack2")?;
            g.stack(
                vec![shared_tuple.tuple_get(0)?, shared_tuple.tuple_get(1)?],
                vec![2],
            )?
            .set_name("stack3")?;
            g.stack(
                vec![shared_tuple.tuple_get(0)?, shared_tuple.tuple_get(0)?],
                vec![2],
            )?
            .set_name("stack4")?;

            let result_classes = generate_equivalence_class(
                c.clone(),
                vec![vec![IOStatus::Shared, IOStatus::Public, IOStatus::Party(1)]],
            )?;

            assert_eq!(
                get_class_from_name(&g, &result_classes, "stack1")?,
                private_class()
            );
            assert_eq!(
                get_class_from_name(&g, &result_classes, "stack2")?,
                public_class()
            );
            assert_eq!(
                get_class_from_name(&g, &result_classes, "stack3")?,
                private_class()
            );
            assert_eq!(
                get_class_from_name(&g, &result_classes, "stack4")?,
                share0_class()
            );

            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_vector_to_array() {
        || -> Result<()> {
            let c = create_unchecked_context()?;
            let g = c.create_graph()?;
            let t = array_type(vec![10], BIT);
            let shared_tuple = g.input(tuple_type(vec![t.clone(), t.clone(), t.clone()]))?;
            let public_array = g.input(t.clone())?;
            let private_array = g.input(t.clone())?;

            let vector1 =
                g.create_vector(t.clone(), vec![public_array.clone(), private_array.clone()])?;
            vector1.vector_to_array()?.set_name("vector_to_array1")?;

            let vector2 =
                g.create_vector(t.clone(), vec![public_array.clone(), public_array.clone()])?;
            vector2.vector_to_array()?.set_name("vector_to_array2")?;

            let vector3 = g.create_vector(
                t.clone(),
                vec![shared_tuple.tuple_get(0)?, shared_tuple.tuple_get(1)?],
            )?;
            vector3.vector_to_array()?.set_name("vector_to_array3")?;

            let vector4 = g.create_vector(
                t.clone(),
                vec![shared_tuple.tuple_get(1)?, shared_tuple.tuple_get(1)?],
            )?;
            vector4.vector_to_array()?.set_name("vector_to_array4")?;

            let result_classes = generate_equivalence_class(
                c.clone(),
                vec![vec![IOStatus::Shared, IOStatus::Public, IOStatus::Party(1)]],
            )?;

            assert_eq!(
                get_class_from_name(&g, &result_classes, "vector_to_array1")?,
                private_class()
            );
            assert_eq!(
                get_class_from_name(&g, &result_classes, "vector_to_array2")?,
                public_class()
            );
            assert_eq!(
                get_class_from_name(&g, &result_classes, "vector_to_array3")?,
                private_class()
            );
            assert_eq!(
                get_class_from_name(&g, &result_classes, "vector_to_array4")?,
                share1_class()
            );

            Ok(())
        }()
        .unwrap();
    }

    #[test]
    fn test_reshape() {
        || -> Result<()> {
            let c = create_unchecked_context()?;
            let g = c.create_graph()?;
            let t = array_type(vec![10], BIT);
            let shared_tuple = g.input(tuple_type(vec![t.clone(), t.clone(), t.clone()]))?;
            let public_array = g.input(t.clone())?;
            let private_array = g.input(t.clone())?;

            shared_tuple
                .reshape(vector_type(3, t.clone()))?
                .set_name("reshape1")?;
            shared_tuple
                .reshape(tuple_type(vec![
                    t.clone(),
                    tuple_type(vec![t.clone(), t.clone()]),
                ]))?
                .set_name("reshape2")?;
            public_array
                .reshape(array_type(vec![2, 5], BIT))?
                .set_name("reshape3")?;
            private_array
                .reshape(array_type(vec![5, 2], BIT))?
                .set_name("reshape4")?;

            let result_classes = generate_equivalence_class(
                c.clone(),
                vec![vec![IOStatus::Shared, IOStatus::Public, IOStatus::Party(1)]],
            )?;

            assert_eq!(
                get_class_from_name(&g, &result_classes, "reshape1")?,
                vector_class(vec![share0_class(), share1_class(), share2_class()])
            );
            assert_eq!(
                get_class_from_name(&g, &result_classes, "reshape2")?,
                vector_class(vec![
                    share0_class(),
                    vector_class(vec![share1_class(), share2_class()])
                ])
            );
            assert_eq!(
                get_class_from_name(&g, &result_classes, "reshape3")?,
                public_class()
            );
            assert_eq!(
                get_class_from_name(&g, &result_classes, "reshape4")?,
                private_class()
            );

            Ok(())
        }()
        .unwrap();
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

        let class_i1 = vector_class(vec![share0_class(), share1_class(), share2_class()]);
        let class_i2 = class_i1.clone();
        let class_i3 = public_class();
        let class_i4 = vector_class(vec![private_class(); 4]);
        let class_i5 = class_i1.clone();
        let class_add_op1 = share0_class();
        let class_add_op2 = share1_class();
        let class_add1 = private_class();
        let class_subtract = private_class();
        let class_multiply = private_class();
        let class_rand1 = private_class();
        let class_rand2 = vector_class(vec![private_class(); 3]);
        let class_nop = class_i3.clone();
        let class_prf1 = vector_class(vec![public_class(); 4]);

        let class_tuple_get1 = share1_class();
        let class_tuple_get2 = private_class();
        let class_create_tuple = vector_class(vec![share1_class(), private_class()]);

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

            let permuteaxes = g.permute_axes(vector_to_array.clone(), vec![0])?;
            permuteaxes.set_name("permuteaxes")?;

            let reshape = g.reshape(permuteaxes.clone(), array_type(vec![2, 2], UINT64))?;
            reshape.set_name("reshape")?;

            let stack = g.stack(vec![reshape.clone(), reshape.clone()], vec![2, 1])?;
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

        let class2_i1 = vector_class(vec![share0_class(), share1_class(), share2_class()]);
        let class2_i2 = class_i1.clone();
        let class2_i3 = public_class();
        let class2_a2b = public_class();
        let class2_b2a = public_class();
        let class2_tuple_get1 = share0_class();
        let class2_tuple_get2 = share1_class();
        let class2_repeat = vector_class(vec![share0_class(); 4]);
        let class2_vector_to_array = share0_class();
        let class2_permuteaxes = share0_class();
        let class2_reshape = share0_class();
        let class2_stack = share0_class();
        let class2_constant = public_class();
        let class2_trunc = share0_class();
        let class2_get_slice = share0_class();
        let class2_array_to_vector = vector_class(vec![share0_class(); 2]);
        let class2_zip = vector_class(vec![class2_array_to_vector.clone(); 2]);
        let class2_vector_get = vector_class(vec![share0_class(); 2]);
        let class2_sum = share0_class();
        let class2_matmul = share0_class();
        let class2_get = share0_class();
        let class2_dot = share0_class();
        let class2_create_named_tuple = vector_class(vec![share0_class(); 2]);
        let class2_named_tuple_get = share0_class();
        let class2_create_vector = vector_class(vec![public_class(); 2]);
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
                        .retrieve_node(context2.retrieve_graph("test_g2").unwrap(), "permuteaxes")
                        .unwrap()
                        .get_global_id()
                )
                .unwrap(),
            class2_permuteaxes
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

            let shared = vector_class(vec![share0_class(), share1_class(), share2_class()]);
            assert_eq!(*test_class1.get(&(0, 0)).unwrap(), private_class());
            assert_eq!(*test_class1.get(&(0, 1)).unwrap(), share0_class());
            assert_eq!(*test_class1.get(&(0, 2)).unwrap(), private_class());
            assert_eq!(*test_class1.get(&(0, 3)).unwrap(), share1_class());
            assert_eq!(*test_class1.get(&(0, 4)).unwrap(), private_class());
            assert_eq!(*test_class1.get(&(0, 5)).unwrap(), share2_class());
            assert_eq!(*test_class1.get(&(0, 6)).unwrap(), shared.clone());
            assert_eq!(*test_class1.get(&(0, 7)).unwrap(), shared.clone());
            assert_eq!(*test_class1.get(&(0, 8)).unwrap(), public_class());
            assert_eq!(*test_class1.get(&(0, 9)).unwrap(), share0_class());
            assert_eq!(*test_class1.get(&(0, 10)).unwrap(), share0_class());
            assert_eq!(*test_class1.get(&(0, 11)).unwrap(), share1_class());
            assert_eq!(*test_class1.get(&(0, 12)).unwrap(), share1_class());
            assert_eq!(*test_class1.get(&(0, 13)).unwrap(), share2_class());
            assert_eq!(*test_class1.get(&(0, 14)).unwrap(), share2_class());
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

            let shared = vector_class(vec![share0_class(), share1_class(), share2_class()]);
            // PRF keys
            assert_eq!(*test_class1.get(&(0, 0)).unwrap(), private_class());
            assert_eq!(*test_class1.get(&(0, 1)).unwrap(), share0_class());
            assert_eq!(*test_class1.get(&(0, 2)).unwrap(), private_class());
            assert_eq!(*test_class1.get(&(0, 3)).unwrap(), share1_class());
            assert_eq!(*test_class1.get(&(0, 4)).unwrap(), private_class());
            assert_eq!(*test_class1.get(&(0, 5)).unwrap(), share2_class());
            // Create PRF triple
            assert_eq!(*test_class1.get(&(0, 6)).unwrap(), shared.clone());
            // Shared input
            assert_eq!(*test_class1.get(&(0, 7)).unwrap(), shared.clone());
            // Public input
            assert_eq!(*test_class1.get(&(0, 8)).unwrap(), public_class());
            // Extract share 0
            assert_eq!(*test_class1.get(&(0, 9)).unwrap(), share0_class());
            // Multiply share 0 by the public value
            assert_eq!(*test_class1.get(&(0, 10)).unwrap(), share0_class());
            // Extract share 1
            assert_eq!(*test_class1.get(&(0, 11)).unwrap(), share1_class());
            // Multiply share 1 by the public value
            assert_eq!(*test_class1.get(&(0, 12)).unwrap(), share1_class());
            // Extract share 2
            assert_eq!(*test_class1.get(&(0, 13)).unwrap(), share2_class());
            // Multiply share 2 by the public value
            assert_eq!(*test_class1.get(&(0, 14)).unwrap(), share2_class());
            // Shared product
            assert_eq!(*test_class1.get(&(0, 15)).unwrap(), shared.clone());
            // Extract shares
            assert_eq!(*test_class1.get(&(0, 16)).unwrap(), share0_class());
            assert_eq!(*test_class1.get(&(0, 17)).unwrap(), share1_class());
            assert_eq!(*test_class1.get(&(0, 18)).unwrap(), share2_class());
            // Revealing
            // Share 2 is sent to party 0, thus becoming public
            assert_eq!(*test_class1.get(&(0, 19)).unwrap(), public_class());
            // Sum of shares 0 and 1 must be private (party 0 has the correct sum)
            assert_eq!(*test_class1.get(&(0, 20)).unwrap(), private_class());
            // Sum of shares 0, 1 and 2 must be private (party 0 has the correct sum)
            assert_eq!(*test_class1.get(&(0, 21)).unwrap(), private_class());
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

            let shared = vector_class(vec![share0_class(), share1_class(), share2_class()]);
            // PRF keys
            assert_eq!(*test_class1.get(&(0, 0)).unwrap(), private_class());
            assert_eq!(*test_class1.get(&(0, 1)).unwrap(), share0_class());
            assert_eq!(*test_class1.get(&(0, 2)).unwrap(), private_class());
            assert_eq!(*test_class1.get(&(0, 3)).unwrap(), share1_class());
            assert_eq!(*test_class1.get(&(0, 4)).unwrap(), private_class());
            assert_eq!(*test_class1.get(&(0, 5)).unwrap(), share2_class());
            // Create PRF triple
            assert_eq!(*test_class1.get(&(0, 6)).unwrap(), shared.clone());
            // Shared input
            assert_eq!(*test_class1.get(&(0, 7)).unwrap(), shared.clone());
            // Public input
            assert_eq!(*test_class1.get(&(0, 8)).unwrap(), public_class());
            // Extract share 0
            assert_eq!(*test_class1.get(&(0, 9)).unwrap(), share0_class());
            // Multiply share 0 by the public value
            assert_eq!(*test_class1.get(&(0, 10)).unwrap(), share0_class());
            // Extract share 1
            assert_eq!(*test_class1.get(&(0, 11)).unwrap(), share1_class());
            // Multiply share 1 by the public value
            assert_eq!(*test_class1.get(&(0, 12)).unwrap(), share1_class());
            // Extract share 2
            assert_eq!(*test_class1.get(&(0, 13)).unwrap(), share2_class());
            // Multiply share 2 by the public value
            assert_eq!(*test_class1.get(&(0, 14)).unwrap(), share2_class());
            // Shared product
            assert_eq!(*test_class1.get(&(0, 15)).unwrap(), shared.clone());
            // Extract shares
            assert_eq!(*test_class1.get(&(0, 16)).unwrap(), share0_class());
            assert_eq!(*test_class1.get(&(0, 17)).unwrap(), share1_class());
            assert_eq!(*test_class1.get(&(0, 18)).unwrap(), share2_class());
            // Revealing
            // Share 2 is sent to party 0, thus becoming public
            assert_eq!(*test_class1.get(&(0, 19)).unwrap(), public_class());
            // Sum of shares 0 and 1 must be private (party 0 has the correct sum)
            assert_eq!(*test_class1.get(&(0, 20)).unwrap(), private_class());
            // Sum of shares 0, 1 and 2 must be private (party 0 has the correct sum)
            assert_eq!(*test_class1.get(&(0, 21)).unwrap(), private_class());
            // Send the revealed value to another party
            assert_eq!(*test_class1.get(&(0, 22)).unwrap(), share0_class());
            // Output node can't have Send annotation
            assert_eq!(*test_class1.get(&(0, 23)).unwrap(), share0_class());
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
