#!/usr/bin/env Rscript
#
# Template R script for benchmarking HarmonizR.
# Called by run_benchmarks.py with arguments:
#
#   Rscript template_run.R <data_tsv> <batch_csv> <output_tsv> <algorithm> <combat_mode> <block> <sort>
#
# Prints R_TIME: <seconds> to stdout as the last line for the harness to parse.

suppressPackageStartupMessages({
    if (requireNamespace("renv", quietly = TRUE)) {
        renv::load()
    }
    library(HarmonizR)
})

args <- commandArgs(trailingOnly = TRUE)
data_path <- args[1]
desc_path <- args[2]
output_path <- args[3]
algorithm <- args[4]

combat_mode <- args[5]
if (is.na(combat_mode) || combat_mode == "NA") combat_mode <- 1L else combat_mode <- as.integer(combat_mode)

block_size <- args[6]
if (is.na(block_size) || block_size == "NA") block_size <- NULL else block_size <- as.integer(block_size)

sort_strategy <- args[7]
if (is.na(sort_strategy) || sort_strategy == "NA") sort_strategy <- FALSE

start_time <- proc.time()[["elapsed"]]

result <- harmonizR(
    data_as_input = data_path,
    description_as_input = desc_path,
    algorithm = algorithm,
    ComBat_mode = combat_mode,
    block = block_size,
    sort = sort_strategy,
    ur = TRUE,
    output_file = FALSE,
    verbosity = 0,
    cores = 16,
)

write.table(result, file = output_path, sep = "\t", quote = FALSE, col.names = NA)

elapsed <- proc.time()[["elapsed"]] - start_time
cat(sprintf("R_TIME: %.4f\n", elapsed))
