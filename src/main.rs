extern crate argparse;
extern crate itertools;
extern crate ordered_float;
extern crate rayon;
extern crate fishers_exact;

// #[cfg(test)]
// mod index;

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
use itertools::Itertools;
use command_line_args::Arguments;
use command_line_args::parse_args_or_exit;
use command_line_args::MaxSupportMode;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
// use std::fs;
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
fn contains_rare_item(
    transaction: &Vec<u32>,
    item_count: &HashMap<u32, u32>,
    max_count: u32,
) -> bool {
    for ref item in transaction.iter() {
        if let Some(&count) = item_count.get(item) {
            if count < max_count {
                return true;
            }
        };
    }
    false
}

fn find_gaussian_max_count(item_count: &HashMap<u32, u32>) -> u32 {
    // Sort item counts by increasing frequency
    let counts = item_count.values().cloned().sorted();
    // max count is the count at the 80% percent mark.
    let index = (counts.len() as f64 * 0.8).round() as usize;
    counts[index]
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
    let max_count = match args.max_support_mode {
        MaxSupportMode::Gaussian => find_gaussian_max_count(&item_count),
        MaxSupportMode::Pareto => 0, // tODO!
    };
    println!("Calculated maximum support as {} / {}.", max_count, num_transactions);

    let mut index: Index = Index::new();
    for mut transaction in TransactionReader::new(&args.input_file_path, &mut itemizer) {
        index.insert(&transaction);
        // Only include transactions which contain at least one rate item.
        if !contains_rare_item(&transaction, &item_count, max_count) {
            continue;
        }

        sort_transaction(&mut transaction, &item_count, SortOrder::Decreasing);
        fptree.insert(&transaction, 1);
    }
    println!(
        "Building initial FPTree took {} seconds.",
        timer.elapsed().as_secs()
    );

    println!("Building lookup table for natural log...");
    let mut ln_table = vec![];
    ln_table.push(0.0);
    for i in 1..num_transactions + 1 {
        ln_table.push((i as f64).ln());
    }

    println!("Starting recursive FPGrowth...");
    let timer = Instant::now();
    let patterns: Vec<ItemSet> = rip_growth(
        &fptree,
        &fptree,
        max_count,
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
