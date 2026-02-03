//! Footprint-based OCPT conformance (control flow, multiplicity, identity).
//! 
//! This follows the Python implementation in
//! `OCPT-Conformance-Checking/src/conformance.py` and its helpers,
//! while reusing Rust abstractions where available. Comments note
//! deviations from the paper where the Python implementation differs.

use std::collections::{HashMap, HashSet};

use crate::conformance::object_centric::object_centric_language_abstraction::OCLanguageAbstraction;
use crate::core::event_data::object_centric::linked_ocel::index_linked_ocel::{
    EventIndex, ObjectIndex,
};
use crate::core::event_data::object_centric::linked_ocel::{IndexLinkedOCEL, LinkedOCELAccess};
use crate::core::process_models::object_centric::ocpt::object_centric_process_tree_struct::{
    IdentityRelationKind, OCPTOperatorType, OCPTNode, OCPT,
};
use crate::core::process_models::object_centric::ocpt::{EventType, ObjectType};

type IdentityPattern = (ObjectType, ObjectType, EventType, EventType);

#[derive(Debug, Clone)]
pub struct FootprintConformance {
    pub control_fitness: f64,
    pub control_precision: f64,
    pub multiplicity_fitness: f64,
    pub multiplicity_precision: f64,
    pub identity_fitness: f64,
    pub identity_precision: f64,
    pub overall_fitness: f64,
    pub overall_precision: f64,
}

pub fn compute_footprint_conformance(locel: &IndexLinkedOCEL, ocpt: &OCPT) -> FootprintConformance {
    let log_abs = OCLanguageAbstraction::create_from_ocel(locel);
    let model_abs = OCLanguageAbstraction::create_from_oc_process_tree(ocpt);

    let log_rel = invert_ob_type_map(&log_abs.related_ev_type_per_ob_type);
    let model_rel = invert_ob_type_map(&model_abs.related_ev_type_per_ob_type);
    let log_div = invert_ob_type_map(&log_abs.divergent_ev_type_per_ob_type);
    let model_div = invert_ob_type_map(&model_abs.divergent_ev_type_per_ob_type);
    let log_con = invert_ob_type_map(&log_abs.convergent_ev_type_per_ob_type);
    let model_con = invert_ob_type_map(&model_abs.convergent_ev_type_per_ob_type);
    let log_def = invert_ob_type_map(&log_abs.deficient_ev_type_per_ob_type);
    let model_def = invert_ob_type_map(&model_abs.deficient_ev_type_per_ob_type);
    let log_opt = invert_ob_type_map(&log_abs.optional_ev_type_per_ob_type);
    let model_opt = invert_ob_type_map(&model_abs.optional_ev_type_per_ob_type);

    let (alphabet, object_types, total_activities) =
        build_pattern_universe(&log_rel, &model_rel, &log_abs, &model_abs);

    let (control_log, control_model) = build_control_patterns(
        &log_abs,
        &model_abs,
        &alphabet,
        &object_types,
    );
    let (multiplicity_log, multiplicity_model) = build_multiplicity_patterns(
        &log_rel,
        &log_div,
        &log_def,
        &log_con,
        &log_opt,
        &model_rel,
        &model_div,
        &model_def,
        &model_con,
        &model_opt,
        &total_activities,
        &object_types,
    );

    let log_ident = compute_log_identity_implications(locel, &log_rel);
    let model_ident = compute_model_identity_implications(ocpt, &model_rel);
    let (identity_log, identity_model) = build_identity_patterns(
        &log_ident,
        &model_ident,
        &log_rel,
        &model_rel,
        &total_activities,
        &object_types,
    );

    let (control_fitness, control_precision) = category_scores(&control_log, &control_model);
    let (multiplicity_fitness, multiplicity_precision) =
        category_scores(&multiplicity_log, &multiplicity_model);
    let (identity_fitness, identity_precision) = category_scores(&identity_log, &identity_model);

    let overall_fitness = (control_fitness + multiplicity_fitness + identity_fitness) / 3.0;
    let overall_precision = (control_precision + multiplicity_precision + identity_precision) / 3.0;

    FootprintConformance {
        control_fitness,
        control_precision,
        multiplicity_fitness,
        multiplicity_precision,
        identity_fitness,
        identity_precision,
        overall_fitness,
        overall_precision,
    }
}

pub fn compute_footprint_conformance_ocpt_vs_ocpt(
    log_ocpt: &OCPT,
    model_ocpt: &OCPT,
) -> FootprintConformance {
    let log_abs = OCLanguageAbstraction::create_from_oc_process_tree(log_ocpt);
    let model_abs = OCLanguageAbstraction::create_from_oc_process_tree(model_ocpt);

    let log_rel = invert_ob_type_map(&log_abs.related_ev_type_per_ob_type);
    let model_rel = invert_ob_type_map(&model_abs.related_ev_type_per_ob_type);
    let log_div = invert_ob_type_map(&log_abs.divergent_ev_type_per_ob_type);
    let model_div = invert_ob_type_map(&model_abs.divergent_ev_type_per_ob_type);
    let log_con = invert_ob_type_map(&log_abs.convergent_ev_type_per_ob_type);
    let model_con = invert_ob_type_map(&model_abs.convergent_ev_type_per_ob_type);
    let log_def = invert_ob_type_map(&log_abs.deficient_ev_type_per_ob_type);
    let model_def = invert_ob_type_map(&model_abs.deficient_ev_type_per_ob_type);
    let log_opt = invert_ob_type_map(&log_abs.optional_ev_type_per_ob_type);
    let model_opt = invert_ob_type_map(&model_abs.optional_ev_type_per_ob_type);

    let (alphabet, object_types, total_activities) =
        build_pattern_universe(&log_rel, &model_rel, &log_abs, &model_abs);

    let (control_log, control_model) =
        build_control_patterns(&log_abs, &model_abs, &alphabet, &object_types);
    let (multiplicity_log, multiplicity_model) = build_multiplicity_patterns(
        &log_rel,
        &log_div,
        &log_def,
        &log_con,
        &log_opt,
        &model_rel,
        &model_div,
        &model_def,
        &model_con,
        &model_opt,
        &total_activities,
        &object_types,
    );

    let log_ident = compute_model_identity_implications(log_ocpt, &log_rel);
    let model_ident = compute_model_identity_implications(model_ocpt, &model_rel);
    let (identity_log, identity_model) = build_identity_patterns(
        &log_ident,
        &model_ident,
        &log_rel,
        &model_rel,
        &total_activities,
        &object_types,
    );

    let (control_fitness, control_precision) = category_scores(&control_log, &control_model);
    let (multiplicity_fitness, multiplicity_precision) =
        category_scores(&multiplicity_log, &multiplicity_model);
    let (identity_fitness, identity_precision) = category_scores(&identity_log, &identity_model);

    let overall_fitness = (control_fitness + multiplicity_fitness + identity_fitness) / 3.0;
    let overall_precision =
        (control_precision + multiplicity_precision + identity_precision) / 3.0;

    FootprintConformance {
        control_fitness,
        control_precision,
        multiplicity_fitness,
        multiplicity_precision,
        identity_fitness,
        identity_precision,
        overall_fitness,
        overall_precision,
    }
}

fn category_scores(log_values: &[bool], model_values: &[bool]) -> (f64, f64) {
    let mut total_log = 0usize;
    let mut total_model = 0usize;
    let mut overlap = 0usize;

    for (log_v, model_v) in log_values.iter().zip(model_values.iter()) {
        if *log_v {
            total_log += 1;
        }
        if *model_v {
            total_model += 1;
        }
        if *log_v && *model_v {
            overlap += 1;
        }
    }

    let fitness = if total_log > 0 {
        overlap as f64 / total_log as f64
    } else {
        1.0
    };
    let precision = if total_model > 0 {
        overlap as f64 / total_model as f64
    } else {
        1.0
    };

    (fitness, precision)
}

fn build_pattern_universe(
    log_rel: &HashMap<EventType, HashSet<ObjectType>>,
    model_rel: &HashMap<EventType, HashSet<ObjectType>>,
    log_abs: &OCLanguageAbstraction,
    model_abs: &OCLanguageAbstraction,
) -> (Vec<EventType>, Vec<ObjectType>, Vec<EventType>) {
    let mut alphabet_set: HashSet<EventType> = log_rel.keys().cloned().collect();
    alphabet_set.extend(model_rel.keys().cloned());
    let mut alphabet: Vec<EventType> = alphabet_set.into_iter().collect();
    alphabet.sort();

    let mut object_type_set: HashSet<ObjectType> = log_abs
        .directly_follows_ev_types_per_ob_type
        .keys()
        .cloned()
        .collect();
    object_type_set.extend(
        model_abs
            .directly_follows_ev_types_per_ob_type
            .keys()
            .cloned(),
    );
    let mut object_types: Vec<ObjectType> = object_type_set.into_iter().collect();
    object_types.sort();

    // NOTE: Python uses a concatenated list here, which can double-count activities.
    // This deviates from the paper, which implies a set-based universe of patterns.
    let mut total_activities: Vec<EventType> = Vec::new();
    total_activities.extend(log_rel.keys().cloned());
    total_activities.extend(model_rel.keys().cloned());
    total_activities.sort();

    (alphabet, object_types, total_activities)
}

fn build_control_patterns(
    log_abs: &OCLanguageAbstraction,
    model_abs: &OCLanguageAbstraction,
    alphabet: &[EventType],
    object_types: &[ObjectType],
) -> (Vec<bool>, Vec<bool>) {
    let mut log_values = Vec::new();
    let mut model_values = Vec::new();

    for a in alphabet {
        for b in alphabet {
            for ot in object_types {
                let log_has = log_abs
                    .directly_follows_ev_types_per_ob_type
                    .get(ot)
                    .is_some_and(|set| set.contains(&(a.clone(), b.clone())));
                let model_has = model_abs
                    .directly_follows_ev_types_per_ob_type
                    .get(ot)
                    .is_some_and(|set| set.contains(&(a.clone(), b.clone())));
                log_values.push(log_has);
                model_values.push(model_has);
            }
        }
    }

    for a in alphabet {
        for ot in object_types {
            let log_has = log_abs
                .start_ev_type_per_ob_type
                .get(ot)
                .is_some_and(|set| set.contains(a));
            let model_has = model_abs
                .start_ev_type_per_ob_type
                .get(ot)
                .is_some_and(|set| set.contains(a));
            log_values.push(log_has);
            model_values.push(model_has);
        }
    }

    for a in alphabet {
        for ot in object_types {
            let log_has = log_abs
                .end_ev_type_per_ob_type
                .get(ot)
                .is_some_and(|set| set.contains(a));
            let model_has = model_abs
                .end_ev_type_per_ob_type
                .get(ot)
                .is_some_and(|set| set.contains(a));
            log_values.push(log_has);
            model_values.push(model_has);
        }
    }

    (log_values, model_values)
}

fn build_multiplicity_patterns(
    log_rel: &HashMap<EventType, HashSet<ObjectType>>,
    log_div: &HashMap<EventType, HashSet<ObjectType>>,
    log_def: &HashMap<EventType, HashSet<ObjectType>>,
    log_con: &HashMap<EventType, HashSet<ObjectType>>,
    log_opt: &HashMap<EventType, HashSet<ObjectType>>,
    model_rel: &HashMap<EventType, HashSet<ObjectType>>,
    model_div: &HashMap<EventType, HashSet<ObjectType>>,
    model_def: &HashMap<EventType, HashSet<ObjectType>>,
    model_con: &HashMap<EventType, HashSet<ObjectType>>,
    model_opt: &HashMap<EventType, HashSet<ObjectType>>,
    total_activities: &[EventType],
    object_types: &[ObjectType],
) -> (Vec<bool>, Vec<bool>) {
    // NOTE: The paper uses multiplicity patterns rel_ot>1, rel_ot<1, rel_a>1, rel_a<1.
    // The Python implementation instead uses related/divergent/deficient/convergent/optional
    // patterns from the abstractions. We mirror the Python behavior here.
    let log_maps = [log_rel, log_div, log_def, log_con, log_opt];
    let model_maps = [model_rel, model_div, model_def, model_con, model_opt];

    let mut log_values = Vec::new();
    let mut model_values = Vec::new();

    for a in total_activities {
        for ot in object_types {
            for i in 0..5usize {
                let log_has = log_maps[i]
                    .get(a)
                    .is_some_and(|set| set.contains(ot));
                let model_has = model_maps[i]
                    .get(a)
                    .is_some_and(|set| set.contains(ot));
                log_values.push(log_has);
                model_values.push(model_has);
            }
        }
    }

    (log_values, model_values)
}

fn build_identity_patterns(
    log_ident: &HashSet<IdentityPattern>,
    model_ident: &HashSet<IdentityPattern>,
    log_rel: &HashMap<EventType, HashSet<ObjectType>>,
    model_rel: &HashMap<EventType, HashSet<ObjectType>>,
    total_activities: &[EventType],
    object_types: &[ObjectType],
) -> (Vec<bool>, Vec<bool>) {
    let mut log_values = Vec::new();
    let mut model_values = Vec::new();

    for a in total_activities {
        for b in total_activities {
            if a == b {
                continue;
            }
            for ot1 in object_types {
                for ot2 in object_types {
                    if ot1 == ot2 {
                        continue;
                    }

                    let rel_ok = related_in_any(a, b, ot1, ot2, log_rel, model_rel);
                    if !rel_ok {
                        continue;
                    }

                    let key = (ot1.clone(), ot2.clone(), a.clone(), b.clone());
                    log_values.push(!log_ident.contains(&key));
                    model_values.push(!model_ident.contains(&key));
                }
            }
        }
    }

    (log_values, model_values)
}

fn related_in_any(
    a: &str,
    b: &str,
    ot1: &str,
    ot2: &str,
    log_rel: &HashMap<EventType, HashSet<ObjectType>>,
    model_rel: &HashMap<EventType, HashSet<ObjectType>>,
) -> bool {
    let log_ok = log_rel
        .get(a)
        .is_some_and(|set| set.contains(ot1) && set.contains(ot2))
        && log_rel
            .get(b)
            .is_some_and(|set| set.contains(ot1) && set.contains(ot2));

    if log_ok {
        return true;
    }

    model_rel
        .get(a)
        .is_some_and(|set| set.contains(ot1) && set.contains(ot2))
        && model_rel
            .get(b)
            .is_some_and(|set| set.contains(ot1) && set.contains(ot2))
}

fn invert_ob_type_map(
    map: &HashMap<ObjectType, HashSet<EventType>>,
) -> HashMap<EventType, HashSet<ObjectType>> {
    let mut result: HashMap<EventType, HashSet<ObjectType>> = HashMap::new();
    for (ob_type, ev_types) in map {
        for ev in ev_types {
            result
                .entry(ev.clone())
                .or_default()
                .insert(ob_type.clone());
        }
    }
    result
}

fn compute_log_identity_implications(
    locel: &IndexLinkedOCEL,
    log_rel: &HashMap<EventType, HashSet<ObjectType>>,
) -> HashSet<IdentityPattern> {
    // NOTE: This mirrors the Python `check_relation_log` logic (hash-based implication checks),
    // which is an implementation choice and not fully specified in the paper.
    let object_types: Vec<ObjectType> = locel.get_ob_types().map(|ot| ot.to_string()).collect();
    let mut events_by_type: HashMap<EventType, Vec<EventIndex>> = HashMap::new();
    for ev_type in locel.get_ev_types() {
        let evs = locel.get_evs_of_type(ev_type).cloned().collect::<Vec<_>>();
        events_by_type.insert(ev_type.to_string(), evs);
    }

    let mut result: HashSet<IdentityPattern> = HashSet::new();

    for ot1 in &object_types {
        for ot2 in &object_types {
            if ot1 == ot2 {
                continue;
            }

            let activities = activities_for_type_pair(log_rel, ot1, ot2);
            for a in &activities {
                for b in &activities {
                    if a == b {
                        continue;
                    }

                    let mut subrelations: Vec<(EventIndex, Vec<ObjectIndex>)> = Vec::new();
                    if let Some(events) = events_by_type.get(a) {
                        subrelations.extend(collect_event_objects_for_types(
                            locel, events, ot1, ot2,
                        ));
                    }
                    if let Some(events) = events_by_type.get(b) {
                        subrelations.extend(collect_event_objects_for_types(
                            locel, events, ot1, ot2,
                        ));
                    }

                    if subrelations.is_empty() {
                        continue;
                    }

                    if check_implication(locel, ot1, &subrelations) {
                        result.insert((ot1.clone(), ot2.clone(), a.clone(), b.clone()));
                    }
                }
            }
        }
    }

    result
}

fn activities_for_type_pair(
    log_rel: &HashMap<EventType, HashSet<ObjectType>>,
    ot1: &str,
    ot2: &str,
) -> Vec<EventType> {
    let mut activities: Vec<EventType> = log_rel
        .iter()
        .filter_map(|(act, types)| {
            if types.contains(ot1) || types.contains(ot2) {
                Some(act.clone())
            } else {
                None
            }
        })
        .collect();
    activities.sort();
    activities
}

fn collect_event_objects_for_types(
    locel: &IndexLinkedOCEL,
    events: &[EventIndex],
    ot1: &str,
    ot2: &str,
) -> Vec<(EventIndex, Vec<ObjectIndex>)> {
    let mut result = Vec::new();
    for ev in events {
        let mut objs: Vec<ObjectIndex> = locel
            .get_e2o_set(ev)
            .iter()
            .filter(|ob| {
                let ob_type = locel.get_ob_type_of(*ob);
                ob_type == ot1 || ob_type == ot2
            })
            .copied()
            .collect();

        if objs.is_empty() {
            continue;
        }

        objs.sort_unstable_by_key(|ob| ob.into_inner());
        result.push((*ev, objs));
    }
    result
}

fn check_implication(
    locel: &IndexLinkedOCEL,
    ot1: &str,
    subrelations: &[(EventIndex, Vec<ObjectIndex>)],
) -> bool {
    let mut object_hashes: HashMap<ObjectIndex, HashSet<Vec<ObjectIndex>>> = HashMap::new();

    for (_ev, objs) in subrelations {
        for ob in objs {
            object_hashes
                .entry(*ob)
                .or_default()
                .insert(objs.clone());
        }
    }

    let max_all = object_hashes
        .values()
        .map(|hashes| hashes.len())
        .max()
        .unwrap_or(0);
    let mut max_ot1 = 0usize;
    for (ob, hashes) in &object_hashes {
        if locel.get_ob_type_of(ob) == ot1 {
            max_ot1 = max_ot1.max(hashes.len());
        }
    }

    let check1 = max_all == 1 && max_all > 0;
    let check2 = max_ot1 == 1 && max_ot1 > 0;
    check1 || check2
}

fn compute_model_identity_implications(
    ocpt: &OCPT,
    model_rel: &HashMap<EventType, HashSet<ObjectType>>,
) -> HashSet<IdentityPattern> {
    let mut raw: HashSet<IdentityPattern> = HashSet::new();
    collect_tree_identity_patterns(&ocpt.root, model_rel, &mut raw);
    close_identity_patterns(raw)
}

fn collect_tree_identity_patterns(
    node: &OCPTNode,
    model_rel: &HashMap<EventType, HashSet<ObjectType>>,
    acc: &mut HashSet<IdentityPattern>,
) {
    match node {
        OCPTNode::Operator(op) => match &op.operator_type {
            OCPTOperatorType::IdentityRelation(rel) => {
                let child = op.children.first();
                let activities = child.map(collect_activities).unwrap_or_default();

                for a in &activities {
                    for b in &activities {
                        if a == b {
                            continue;
                        }
                        for left in &rel.left {
                            for right in &rel.right {
                                if !model_rel
                                    .get(a)
                                    .is_some_and(|set| set.contains(left) && set.contains(right))
                                {
                                    continue;
                                }
                                if !model_rel
                                    .get(b)
                                    .is_some_and(|set| set.contains(left) && set.contains(right))
                                {
                                    continue;
                                }

                                acc.insert((left.clone(), right.clone(), a.clone(), b.clone()));
                                if rel.kind == IdentityRelationKind::Sync {
                                    acc.insert((right.clone(), left.clone(), a.clone(), b.clone()));
                                }
                            }
                        }
                    }
                }

                if let Some(child) = child {
                    collect_tree_identity_patterns(child, model_rel, acc);
                }
            }
            _ => {
                for child in &op.children {
                    collect_tree_identity_patterns(child, model_rel, acc);
                }
            }
        },
        OCPTNode::Leaf(_) => {}
    }
}

fn collect_activities(node: &OCPTNode) -> HashSet<EventType> {
    let mut result = HashSet::new();
    match node {
        OCPTNode::Operator(op) => {
            for child in &op.children {
                result.extend(collect_activities(child));
            }
        }
        OCPTNode::Leaf(leaf) => {
            if let crate::core::process_models::object_centric::ocpt::object_centric_process_tree_struct::OCPTLeafLabel::Activity(label) =
                &leaf.activity_label
            {
                result.insert(label.clone());
            }
        }
    }
    result
}

fn close_identity_patterns(
    raw: HashSet<IdentityPattern>,
) -> HashSet<IdentityPattern> {
    let mut by_activity: HashMap<(EventType, EventType), HashSet<(ObjectType, ObjectType)>> =
        HashMap::new();

    for (ot1, ot2, a, b) in raw {
        by_activity
            .entry((a, b))
            .or_default()
            .insert((ot1, ot2));
    }

    let mut result: HashSet<IdentityPattern> = HashSet::new();
    for ((a, b), edges) in by_activity {
        let closure = transitive_closure(edges);
        for (ot1, ot2) in closure {
            result.insert((ot1, ot2, a.clone(), b.clone()));
        }
    }

    result
}

fn transitive_closure(edges: HashSet<(ObjectType, ObjectType)>) -> HashSet<(ObjectType, ObjectType)> {
    let mut adj: HashMap<ObjectType, HashSet<ObjectType>> = HashMap::new();
    let mut nodes: HashSet<ObjectType> = HashSet::new();
    for (from, to) in &edges {
        adj.entry(from.clone()).or_default().insert(to.clone());
        nodes.insert(from.clone());
        nodes.insert(to.clone());
    }

    let mut closure: HashSet<(ObjectType, ObjectType)> = HashSet::new();
    for start in &nodes {
        let mut stack = Vec::new();
        let mut seen: HashSet<ObjectType> = HashSet::new();
        if let Some(nexts) = adj.get(start) {
            stack.extend(nexts.iter().cloned());
        }
        while let Some(curr) = stack.pop() {
            if !seen.insert(curr.clone()) {
                continue;
            }
            closure.insert((start.clone(), curr.clone()));
            if let Some(nexts) = adj.get(&curr) {
                for next in nexts {
                    if !seen.contains(next) {
                        stack.push(next.clone());
                    }
                }
            }
        }
    }
    closure
}
