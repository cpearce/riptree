use std::env;
use std::process;
use std::io;

use argparse::{ArgumentParser, Store};

pub enum MaxSupportMode {
    Pareto,
    Gaussian,
}

pub struct Arguments {
    pub input_file_path: String,
    pub output_rules_path: String,
    pub max_support_mode: MaxSupportMode,
    pub min_confidence: f64,
    pub min_lift: f64,
}

pub fn parse_args_or_exit() -> Arguments {
    let mut args: Arguments = Arguments {
        input_file_path: String::new(),
        output_rules_path: String::new(),
        max_support_mode: MaxSupportMode::Gaussian,
        min_confidence: 0.0,
        min_lift: 0.0,
    };

    let mut max_support_mode: String = String::new();
    {
        let mut parser = ArgumentParser::new();
        parser.set_description("Rare Infrequent Pattern Tree association rule data miner.");

        parser
            .refer(&mut args.input_file_path)
            .add_option(&["--input"], Store, "Input dataset in CSV format.")
            .metavar("file_path")
            .required();

        parser
            .refer(&mut args.output_rules_path)
            .add_option(
                &["--output"],
                Store,
                "File path in which to store output rules. \
                 Format: antecedent -> consequent, confidence, lift, support.",
            )
            .metavar("file_path")
            .required();

        parser
            .refer(&mut max_support_mode)
            .add_option(
                &["--max-support"],
                Store,
                "Method to use to calculate maximum support, either 'gaussian' or 'pareto'",
            )
            .required();

        parser
            .refer(&mut args.min_confidence)
            .add_option(
                &["--min-confidence"],
                Store,
                "Minimum rule confidence threshold, in range [0,1].",
            )
            .metavar("threshold")
            .required();

        parser
            .refer(&mut args.min_lift)
            .add_option(
                &["--min-lift"],
                Store,
                "Minimum rule lift confidence threshold, in range [1,∞].",
            )
            .metavar("threshold");

        if env::args().count() == 1 {
            parser.print_help("Usage:", &mut io::stderr()).unwrap();
            process::exit(1);
        }

        match parser.parse_args() {
            Ok(()) => {}
            Err(err) => {
                process::exit(err);
            }
        }
    }

    args.max_support_mode = match max_support_mode.as_ref() {
        "gaussian" => MaxSupportMode::Gaussian,
        "pareto" => MaxSupportMode::Pareto,
        _ => {
            eprintln!("Error: --max-support-mode must be either 'gaussian' or 'pareto'");
            process::exit(1);
        }
    };

    if args.min_confidence < 0.0 || args.min_confidence > 1.0 {
        eprintln!("Minimum rule confidence threshold must be in range [0,1]");
        process::exit(1);
    }

    if args.min_lift < 1.0 {
        eprintln!("Minimum lift must be in range [1,∞]");
        process::exit(1);
    }

    args
}
