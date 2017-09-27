extern crate argparse;
extern crate itertools;
extern crate ordered_float;
extern crate rand;
extern crate rayon;

mod itemizer;
mod transaction_reader;
mod fptree;
mod generate_rules;
mod command_line_args;
mod index;

use index::Index;
use itemizer::Itemizer;
use transaction_reader::TransactionReader;
use fptree::FPTree;
use fptree::sort_transaction;
use fptree::rip_growth;
use fptree::SortOrder;
use fptree::ItemSet;
use generate_rules::generate_rules;
use generate_rules::Rule;
use command_line_args::Arguments;
use command_line_args::parse_args_or_exit;
use command_line_args::MaxSupportMode;
use rand::Rng;
use rayon::prelude::*;
use std::cmp::{max, min};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::process;
use std::time::Instant;

fn count_item_frequencies(
    reader: TransactionReader,
) -> Result<(HashMap<u32, u32>, usize), Box<Error>> {
    let mut item_count: HashMap<u32, u32> = HashMap::new();
    let mut num_transactions = 0;
    for transaction in reader {
        num_transactions += 1;
        for item in transaction {
            let counter = item_count.entry(item).or_insert(0);
            *counter += 1;
        }
    }
    Ok((item_count, num_transactions))
}

// Returns true if transaction contains at least one rate item.
fn contains_rare_item(transaction: &Vec<u32>, rare_items: &HashSet<u32>) -> bool {
    transaction.iter().any(|item| rare_items.contains(item))
}

fn find_gaussian_rare_items(
    item_count: &HashMap<u32, u32>,
    num_transactions: usize,
    max_item_id: u32,
) -> HashSet<u32> {
    let avg_transaction_len = (item_count.iter().fold(0, |acc, (_, count)| acc + count) as f64 /
        num_transactions as f64)
        .ceil() as u32;

    let max_item_count = item_count
        .iter()
        .fold(0, |acc, (_, count)| max(acc, *count));

    let delta = 0.05;
    let epsilon = ((max_item_count as f64).powi(2) * (1.0_f64 / delta).ln() /
        (2.0 * num_transactions as f64))
        .sqrt();

    // Generate 100 randomly distributed datasets in parallel,
    // and reduce into the minimum count of each item over all datasets.
    let min_count: HashMap<u32, u32> = (0..100)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            let mut random_dataset = HashMap::new();
            for _ in 0..num_transactions {
                for _ in 0..avg_transaction_len {
                    let random_item = rng.gen_range(0, max_item_id + 1);
                    *random_dataset.entry(random_item).or_insert(0) += 1;
                }
            }
            random_dataset
        })
        .reduce(HashMap::new, |mut min_count, random_dataset| {
            for (item, count) in random_dataset.iter() {
                let p = min_count.entry(*item).or_insert(*count);
                *p = min(*p, *count);
            }
            min_count
        });

    // See if the count in the actual dataset is significantly different from
    // the random datasets.
    let mut rare_items: HashSet<u32> = HashSet::new();
    for (item, count) in item_count.iter() {
        let random_min_count = min_count[item] as f64;
        if (random_min_count - (*count as f64)) > epsilon {
            rare_items.insert(*item);
        }
    }

    rare_items
}

fn find_pareto_rare_items(item_count: &HashMap<u32, u32>) -> HashSet<u32> {
    // Sort (item, count) pairs by increasing frequency, and accumulate the
    // total sum of the counts of all items.
    let mut item_count_sum = 0;
    let mut items = Vec::with_capacity(item_count.len());
    for (&item, &count) in item_count.iter() {
        item_count_sum += count;
        items.push((item, count));
    }
    items.sort_by(|&(_, a), &(_, b)| a.cmp(&b));

    let threshold = (0.25 * item_count_sum as f64) as u32;
    let mut rare_items: HashSet<u32> = HashSet::new();
    let mut sum = 0;
    let mut prev_count = 0;
    for (item, count) in items {
        sum += count;
        // If this item as the same count as the previous, include it.
        // This ensures that all items of the same count are included
        // if any are included, otherwise, the order in which items are
        // iterated here is significant in the results, i.e. they're
        // non-deterministic.
        if sum < threshold || prev_count == count {
            rare_items.insert(item);
        }
        if sum > threshold && prev_count != count {
            break;
        }
        prev_count = count;
    }

    rare_items
}

fn mine_rip_tree(args: &Arguments) -> Result<(), Box<Error>> {
    println!("Mining data set: {}", args.input_file_path);
    println!("Making first pass of dataset to count item frequencies...");
    // Make one pass of the dataset to calculate the item frequencies
    // for the initial tree.
    let start = Instant::now();
    let timer = Instant::now();
    let mut itemizer: Itemizer = Itemizer::new();
    let (item_count, num_transactions) = count_item_frequencies(
        TransactionReader::new(&args.input_file_path, &mut itemizer),
    ).unwrap();
    println!(
        "First pass took {} seconds, num_transactions={}.",
        timer.elapsed().as_secs(),
        num_transactions
    );

    println!("Building initial RIPTree based on item frequencies...");

    // Load the initial tree, by re-reading the data set and inserting
    // each transaction into the tree sorted by item frequency.
    let timer = Instant::now();
    let mut fptree = FPTree::new();
    let rare_items = match args.max_support_mode {
        MaxSupportMode::Gaussian => {
            find_gaussian_rare_items(&item_count, num_transactions, itemizer.max_item_id())
        }
        MaxSupportMode::Pareto => find_pareto_rare_items(&item_count),
    };
    assert!(rare_items.len() > 0);
    println!(
        "{} of {} items are considered rare.",
        rare_items.len(),
        item_count.len()
    );

    let mut index: Index = Index::new();
    for mut transaction in TransactionReader::new(&args.input_file_path, &mut itemizer) {
        index.insert(&transaction);
        // Only include transactions which contain at least one rate item.
        if !contains_rare_item(&transaction, &rare_items) {
            continue;
        }

        sort_transaction(&mut transaction, &item_count, SortOrder::Decreasing);
        fptree.insert(&transaction, 1);
    }
    println!(
        "Building initial FPTree took {} seconds.",
        timer.elapsed().as_secs()
    );

    println!("Building lookup table for natural log/factorial...");
    let mut ln_table = vec![];
    ln_table.push(0.0);
    ln_table.push(0.0);
    for i in 2..num_transactions + 1 {
        let prev = ln_table[i - 1];
        ln_table.push(prev + (i as f64).ln());
    }

    println!("Starting recursive FPGrowth...");
    let timer = Instant::now();
    let patterns: Vec<ItemSet> = rip_growth(
        &fptree,
        &fptree,
        Some(&rare_items),
        &vec![],
        num_transactions as u32,
        &itemizer,
        &index,
        &ln_table,
    );

    println!(
        "FPGrowth generated {} frequent itemsets in {} seconds.",
        patterns.len(),
        timer.elapsed().as_secs()
    );

    println!("Generating rules...");
    let timer = Instant::now();
    let rules: Vec<Rule> = generate_rules(
        &patterns,
        num_transactions as u32,
        args.min_confidence,
        args.min_lift,
        &rare_items,
        &index,
        &ln_table,
        &item_count,
        args.disable_family_wise_rule_filtering,
        args.disable_permutation_rule_filtering,
    ).iter()
        .cloned()
        .collect();
    println!(
        "Generated {} rules in {} seconds, writing to disk.",
        rules.len(),
        timer.elapsed().as_secs()
    );

    let timer = Instant::now();
    {
        let mut output = BufWriter::new(File::create(&args.output_rules_path).unwrap());
        writeln!(
            output,
            "Antecedent => Consequent, Confidence, Lift, Support"
        )?;
        for rule in rules {
            writeln!(
                output,
                "{}, {}, {}, {}",
                rule.to_string(&itemizer),
                rule.confidence(),
                rule.lift(),
                rule.support(),
            )?;
        }
    }
    println!(
        "Wrote rules to disk in {} seconds.",
        timer.elapsed().as_secs()
    );

    println!("Total runtime: {} seconds", start.elapsed().as_secs());

    Ok(())
}

fn main() {
    let arguments = parse_args_or_exit();

    if let Err(err) = mine_rip_tree(&arguments) {
        println!("Error: {}", err);
        process::exit(1);
    }
}
