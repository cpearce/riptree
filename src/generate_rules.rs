use itemizer::Itemizer;
use itertools::Itertools;
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use std::collections::HashSet;
use std::collections::HashMap;
use fptree::ItemSet;

#[derive(Clone, Hash, Eq, Debug)]
pub struct Rule {
    antecedent: Vec<u32>,
    consequent: Vec<u32>,
    confidence: OrderedFloat<f64>,
    lift: OrderedFloat<f64>,
    support: OrderedFloat<f64>,
}

impl PartialEq for Rule {
    fn eq(&self, other: &Rule) -> bool {
        self.antecedent == other.antecedent && self.consequent == other.consequent
    }
}

// Assumes both itemsets are sorted.
fn union(a: &Vec<u32>, b: &Vec<u32>) -> Vec<u32> {
    let mut c: Vec<u32> = Vec::new();
    let mut ap = 0;
    let mut bp = 0;
    while ap < a.len() && bp < b.len() {
        if a[ap] < b[bp] {
            c.push(a[ap]);
            ap += 1;
        } else if b[bp] < a[ap] {
            c.push(b[bp]);
            bp += 1;
        } else {
            // a[ap] == b[bp]
            c.push(a[ap]);
            ap += 1;
            bp += 1;
        }
    }
    while ap < a.len() {
        c.push(a[ap]);
        ap += 1;
    }
    while bp < b.len() {
        c.push(b[bp]);
        bp += 1;
    }
    c
}

// Assumes both itemsets are sorted.
fn intersection(a: &Vec<u32>, b: &Vec<u32>) -> Vec<u32> {
    let mut c: Vec<u32> = Vec::new();
    let mut ap = 0;
    let mut bp = 0;
    while ap < a.len() && bp < b.len() {
        if a[ap] < b[bp] {
            ap += 1;
        } else if b[bp] < a[ap] {
            bp += 1;
        } else {
            // a[ap] == b[bp]
            c.push(a[ap]);
            ap += 1;
            bp += 1;
        }
    }
    c
}

// If all items in the itemset convert to an integer, order by that integer,
// otherwise order lexicographically.
fn ensure_sorted(a: &mut Vec<String>) {
    let all_items_convert_to_ints = a.iter().all(|ref x| match x.parse::<u32>() {
        Ok(_) => true,
        Err(_) => false,
    });
    if all_items_convert_to_ints {
        a.sort_by(|ref x, ref y| {
            let _x = match x.parse::<u32>() {
                Ok(i) => i,
                Err(_) => 0,
            };
            let _y = match y.parse::<u32>() {
                Ok(i) => i,
                Err(_) => 0,
            };
            _x.cmp(&_y)
        });
    } else {
        a.sort();
    }
}

impl Rule {
    pub fn to_string(&self, itemizer: &Itemizer) -> String {
        let mut a: Vec<String> = self.antecedent
            .iter()
            .map(|&id| itemizer.str_of(id))
            .collect();
        ensure_sorted(&mut a);
        let mut b: Vec<String> = self.consequent
            .iter()
            .map(|&id| itemizer.str_of(id))
            .collect();
        ensure_sorted(&mut b);
        [a.join(" "), " ==> ".to_owned(), b.join(" ")].join("")
    }

    // Creates a new Rule from (antecedent,consequent) if the rule
    // would be above the min_confidence threshold.
    fn make(
        antecedent: Vec<u32>,
        consequent: Vec<u32>,
        itemset_support: &HashMap<Vec<u32>, f64>,
        min_confidence: f64,
        min_lift: f64,
    ) -> Option<Rule> {
        if antecedent.is_empty() || consequent.is_empty() {
            return None;
        }

        let ac_vec: Vec<u32> = union(&antecedent, &consequent);
        let ac_sup = match itemset_support.get(&ac_vec) {
            Some(support) => support.clone(),
            None => return None,
        };

        let a_sup = match itemset_support.get(&antecedent) {
            Some(support) => support.clone(),
            None => return None,
        };

        let confidence = ac_sup / a_sup;
        if confidence < min_confidence {
            return None;
        }
        let c_sup = match itemset_support.get(&consequent) {
            Some(support) => support.clone(),
            None => return None,
        };

        let lift = ac_sup / (a_sup * c_sup);
        if lift < min_lift {
            return None;
        }

        // Note: We sort the antecedent and consequent so that equality
        // tests are consistent.
        Some(Rule {
            antecedent: antecedent.iter().cloned().sorted(),
            consequent: consequent.iter().cloned().sorted(),
            confidence: OrderedFloat::from(confidence),
            lift: OrderedFloat::from(lift),
            support: OrderedFloat::from(ac_sup),
        })
    }

    // Creates a new Rule with:
    //  - the antecedent is the union of both rules' antecedents, and
    //  - the consequent is the intersection of both rules' consequents,
    // provided the new rule would be would be above the min_confidence threshold.
    fn merge(
        a: &Rule,
        b: &Rule,
        itemset_support: &HashMap<Vec<u32>, f64>,
        min_confidence: f64,
        min_lift: f64,
    ) -> Option<Rule> {
        // let antecedent = union(&a.antecedent, &b.antecedent);
        // let consequent = intersection(&a.consequent, &b.consequent);
        let antecedent = intersection(&a.antecedent, &b.antecedent);
        let consequent = union(&a.consequent, &b.consequent);
        Rule::make(
            antecedent,
            consequent,
            itemset_support,
            min_confidence,
            min_lift,
        )
    }

    pub fn confidence(&self) -> f64 {
        self.confidence.into()
    }

    pub fn lift(&self) -> f64 {
        self.lift.into()
    }

    pub fn support(&self) -> f64 {
        self.support.into()
    }

    pub fn union_size(&self) -> usize {
        self.antecedent.len() + self.consequent.len()
    }
}

pub fn split_out_item(items: &Vec<u32>, item: u32) -> (Vec<u32>, Vec<u32>) {
    let antecedent: Vec<u32> = items.iter().filter(|&&x| x != item).cloned().collect();
    let consequent: Vec<u32> = vec![item];
    (antecedent, consequent)
}

pub fn generate_rules(
    itemsets: &Vec<ItemSet>,
    dataset_size: u32,
    min_confidence: f64,
    min_lift: f64,
) -> HashSet<Rule> {
    // Create a lookup of itemset to support, so we can quickly determine
    // an itemset's support during rule generation.
    let mut itemset_support: HashMap<Vec<u32>, f64> = HashMap::with_capacity(itemsets.len());
    for ref i in itemsets.iter() {
        itemset_support.insert(i.items.clone(), i.count as f64 / dataset_size as f64);
    }

    let rv: Vec<HashSet<Rule>> = itemsets
        .par_iter()
        .filter(|i| i.items.len() > 1)
        .map(|ref itemset| {
            let mut rules: HashSet<Rule> = HashSet::new();
            let mut candidates: Vec<Rule> = Vec::new();
            // First level candidates are all the rules with consequents of size 1.
            for &item in itemset.items.iter() {
                let (antecedent, consequent) = split_out_item(&itemset.items, item);
                if let Some(rule) = Rule::make(
                    antecedent,
                    consequent,
                    &itemset_support,
                    min_confidence,
                    min_lift,
                ) {
                    // Passes confidence and lift threshold, keep rule.
                    assert!(!candidates.contains(&rule));
                    assert!(!rules.contains(&rule));
                    candidates.push(rule.clone());
                    rules.insert(rule);
                }
            }

            while !candidates.is_empty() {
                // Subsequent level candidates have their antecedents as the
                // intersection and the consequent as the union of two parent
                // rules.
                let mut next_candidates = vec![];
                for (candidate, other) in candidates.iter().tuple_combinations() {
                    assert_eq!(candidate.union_size(), other.union_size());
                    if let Some(rule) = Rule::merge(
                        &candidate,
                        &other,
                        &itemset_support,
                        min_confidence,
                        min_lift,
                    ) {
                        if !rules.contains(&rule) {
                            rules.insert(rule.clone());
                            next_candidates.push(rule);
                        }
                    }
                }
                // Copy the current generation into the candidates list, so that we
                // use it to calculate the next generation.
                candidates = next_candidates.iter().cloned().collect();

                next_candidates.clear();
            }
            rules
        })
        .collect();

    let mut rules: HashSet<Rule> = HashSet::new();
    for set in rv.into_iter() {
        for rule in set {
            rules.insert(rule);
        }
    }

    rules
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_index() {
        use index::Index;
        use super::ItemSet;
        use super::Itemizer;
        use std::collections::HashMap;

        // HARM's census2.csv test dataset.

        // Load entire dataset into index.
        let mut index = Index::new();
        let transactions = vec![
            vec!["a", "b", "c"],
            vec!["d", "b", "c"],
            vec!["a", "b", "e"],
            vec!["f", "g", "c"],
            vec!["d", "g", "e"],
            vec!["f", "b", "c"],
            vec!["f", "b", "c"],
            vec!["a", "b", "e"],
            vec!["a", "b", "c"],
            vec!["a", "b", "e"],
            vec!["a", "b", "e"],
        ];
        let mut itemizer: Itemizer = Itemizer::new();
        for line in &transactions {
            let transaction = line.iter().map(|s| itemizer.id_of(s)).collect::<Vec<u32>>();
            index.insert(&transaction);
        }

        let itemsets = [
            vec!["b", "e"],
            vec!["a", "e"],
            vec!["a", "b", "e"],
            vec!["f"],
            vec!["c", "f"],
            vec!["b", "f"],
            vec!["b", "c", "f"],
            vec!["g"],
            vec!["a"],
            vec!["a", "b"],
            vec!["b"],
            vec!["c"],
            vec!["b", "c"],
            vec!["c", "g"],
            vec!["d", "g"],
            vec!["d", "e", "g"],
            vec!["e", "g"],
            vec!["f", "g"],
            vec!["c", "f", "g"],
            vec!["a", "c"],
            vec!["a", "b", "c"],
            vec!["d"],
            vec!["b", "d"],
            vec!["c", "d"],
            vec!["b", "c", "d"],
            vec!["d", "e"],
            vec!["e"],
        ].iter()
            .map(|s| itemizer.to_id_vec(s))
            .map(|i| {
                ItemSet::new(
                    i.clone(),
                    (index.support(&i) * transactions.len() as f64) as u32,
                )
            })
            .collect::<Vec<ItemSet>>();

        let rules = super::generate_rules(&itemsets, transactions.len() as u32, 0.05, 1.0);

        let mut expected_rules: HashMap<&str, u32> = [
            ("a ==> b", 0),
            ("a ==> b e", 0),
            ("a ==> e", 0),
            ("a b ==> e", 0),
            ("a c ==> b", 0),
            ("a e ==> b", 0),
            ("b ==> a", 0),
            ("b ==> a e", 0),
            ("b ==> c", 0),
            ("b ==> c d", 0),
            ("b c ==> d", 0),
            ("b c ==> f", 0),
            ("b d ==> c", 0),
            ("b e ==> a", 0),
            ("b f ==> c", 0),
            ("c ==> b", 0),
            ("c ==> b d", 0),
            ("c ==> f", 0),
            ("c ==> f g", 0),
            ("c d ==> b", 0),
            ("c f ==> g", 0),
            ("c g ==> f", 0),
            ("d ==> b c", 0),
            ("d ==> e", 0),
            ("d ==> e g", 0),
            ("d ==> g", 0),
            ("d e ==> g", 0),
            ("d g ==> e", 0),
            ("e ==> a", 0),
            ("e ==> a b", 0),
            ("e ==> d", 0),
            ("e ==> d g", 0),
            ("e ==> g", 0),
            ("e g ==> d", 0),
            ("f ==> c", 0),
            ("f ==> c g", 0),
            ("f ==> g", 0),
            ("f g ==> c", 0),
            ("g ==> c f", 0),
            ("g ==> d", 0),
            ("g ==> d e", 0),
            ("g ==> e", 0),
            ("g ==> f", 0),
        ].iter()
            .cloned()
            .collect();

        for rule_str in rules.iter().map(|r| r.to_string(&itemizer)) {
            assert_eq!(expected_rules.contains_key::<str>(&rule_str), true);
            if let Some(count) = expected_rules.get_mut::<str>(&rule_str) {
                *count += 1;
            }
        }

        for count in expected_rules.values() {
            assert_eq!(*count, 1);
        }
    }
}
