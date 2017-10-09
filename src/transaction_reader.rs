use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use itemizer::Itemizer;
use std::collections::HashSet;

pub struct TransactionReader<'a> {
    reader: BufReader<File>,
    itemizer: &'a mut Itemizer,
}

impl<'a> TransactionReader<'a> {
    pub fn new(path: &str, itemizer: &'a mut Itemizer) -> TransactionReader<'a> {
        let file = File::open(path).unwrap();
        let reader = BufReader::new(file);
        TransactionReader {
            reader: reader,
            itemizer,
        }
    }
}

impl<'a> Iterator for TransactionReader<'a> {
    type Item = Vec<u32>;
    fn next(&mut self) -> Option<Vec<u32>> {
        let mut line = String::new();
        loop {
            let len = self.reader.read_line(&mut line).unwrap();
            if len == 0 {
                return None;
            }
            let splits = line.split(",")
                .map(|s| self.itemizer.id_of(s.trim()))
                .collect::<HashSet<u32>>()
                .iter()
                .cloned()
                .collect::<Vec<u32>>();
            if splits.len() > 0 {
                return Some(splits);
            }
        }
    }
}
