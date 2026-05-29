#!/usr/bin/env Rscript
#
# benchmark_r.R Single-session R benchmark script for HarmonizR.
#
# Reads a JSON scenarios file, runs all scenarios in one R session
# (libraries load once), writes results and concordance outputs.
#
# Usage:
#   Rscript benchmarks/r/benchmark_r.R \
#       --scenarios benchmarks/results/tmp/r_scenarios.json \
#       --output    benchmarks/results/tmp/r_results.json
#
# The scenarios JSON is an array of objects, each with fields:
#   id, data, desc, output_tsv, algorithm, combat_mode, block,
#   sort, n_reps, cores, timeout_s
#
# The output JSON is a single object with:
#   r_version, harmonizr_version, startup_s, results (array)
#
# Per-scenario crash resilience: partial JSON is written after each
# scenario so that a crash mid-batch does not lose all results.

`%||%` <- function(a, b) if (is.null(a)) b else a

suppressPackageStartupMessages({
    if (requireNamespace("renv", quietly = TRUE)) renv::load()
    library(HarmonizR)
})

# ---- Optional timeout support -------------------------------------------
HAS_TIMEOUT <- requireNamespace("R.utils", quietly = TRUE)
if (!HAS_TIMEOUT) {
    message("R.utils not installed; timeouts disabled.")
}

# ---- Argument parsing ---------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
scenarios_path <- NULL
output_path <- NULL

i <- 1
while (i <= length(args)) {
    if (args[i] == "--scenarios" && i < length(args)) {
        scenarios_path <- args[i + 1]
        i <- i + 2
    } else if (args[i] == "--output" && i < length(args)) {
        output_path <- args[i + 1]
        i <- i + 2
    } else {
        stop(sprintf("Unknown argument: %s", args[i]))
    }
}

if (is.null(scenarios_path)) stop("--scenarios is required")
if (is.null(output_path))    stop("--output is required")

# ---- Helpers ------------------------------------------------------------

read_vmrss <- function() {
    tryCatch({
        lines <- readLines("/proc/self/status", warn = FALSE)
        rss_line <- grep("^VmRSS:", lines, value = TRUE)
        as.integer(strsplit(rss_line, "\\s+")[[1]][2])
    }, error = function(e) NA_integer_)
}

# Clean features that would cause singular design matrices (all non-NA
# values in a single batch). These crash sva::ComBat because solve()
# cannot invert a rank-deficient crossprod matrix, whereas Python's
# lstsq handles them gracefully.
clean_singular_features <- function(data_path, desc_path) {
    raw <- read.table(data_path, sep = "\t", header = TRUE, row.names = 1)
    desc <- read.csv(desc_path)
    batch <- desc[, 3]
    design <- model.matrix(~0 + as.factor(batch))
    keep <- rep(TRUE, nrow(raw))
    for (i in seq_len(nrow(raw))) {
        des <- design[!is.na(as.numeric(raw[i, ])), ]
        if (qr(des)$rank < ncol(des)) {
            keep[i] <- FALSE
        }
    }
    n_dropped <- sum(!keep)
    if (n_dropped > 0) {
        message(sprintf("  Dropped %d feature(s) with singular design matrix", n_dropped))
    }
    raw[keep, ]
}

harmonizr_call <- function(scenario) {
    # Convert JSON nulls to R-appropriate values
    algo <- scenario$algorithm

    cm <- scenario$combat_mode
    if (is.null(cm) || is.na(cm)) cm <- 1L else cm <- as.integer(cm)

    blk <- scenario$block
    if (is.null(blk) || is.na(blk)) blk <- NULL else blk <- as.numeric(blk)

    srt <- scenario$sort
    if (is.null(srt) || is.na(srt)) srt <- FALSE

    cores <- scenario$cores
    if (is.null(cores) || is.na(cores)) cores <- 1 else cores <- as.numeric(cores)

    timeout_s <- scenario$timeout_s
    if (is.null(timeout_s) || is.na(timeout_s)) timeout_s <- 300

    result <- tryCatch({
        if (HAS_TIMEOUT) {
            R.utils::withTimeout(
                harmonizR(
                    data_as_input        = scenario$data,
                    description_as_input = scenario$desc,
                    algorithm            = algo,
                    ComBat_mode          = cm,
                    block                = blk,
                    sort                 = srt,
                    ur                   = TRUE,
                    output_file          = FALSE,
                    verbosity            = 0,
                    cores                = cores
                ),
                timeout = timeout_s,
                onTimeout = "error"
            )
        } else {
            harmonizR(
                data_as_input        = scenario$data,
                description_as_input = scenario$desc,
                algorithm            = algo,
                ComBat_mode          = cm,
                block                = blk,
                sort                 = srt,
                ur                   = TRUE,
                output_file          = FALSE,
                verbosity            = 0,
                cores                = cores
            )
        }
    }, error = function(e) {
        msg <- conditionMessage(e)
        # Check if this is a singular matrix error from sva::ComBat
        if (grepl("singular", msg)) {
            message("  sva::ComBat crashed on singular matrix. Retrying with problematic features removed...")
            cleaned <- clean_singular_features(scenario$data, scenario$desc)
            # If we dropped no features but it still crashed, return the error
            if (nrow(cleaned) == 0) {
                return(list(error = msg))
            }
            desc <- read.csv(scenario$desc)
            if (HAS_TIMEOUT) {
                R.utils::withTimeout(
                    harmonizR(
                        data_as_input        = cleaned,
                        description_as_input = desc,
                        algorithm            = algo,
                        ComBat_mode          = cm,
                        block                = blk,
                        sort                 = srt,
                        ur                   = TRUE,
                        output_file          = FALSE,
                        verbosity            = 0,
                        cores                = cores
                    ),
                    timeout = timeout_s,
                    onTimeout = "error"
                )
            } else {
                harmonizR(
                    data_as_input        = cleaned,
                    description_as_input = desc,
                    algorithm            = algo,
                    ComBat_mode          = cm,
                    block                = blk,
                    sort                 = srt,
                    ur                   = TRUE,
                    output_file          = FALSE,
                    verbosity            = 0,
                    cores                = cores
                )
            }
        } else {
            list(error = msg)
        }
    })

    result
}

write_partial_json <- function(results, path, max_rss_kb) {
    out <- list(
        r_version         = R.version.string,
        harmonizr_version = as.character(packageVersion("HarmonizR")),
        startup_s         = startup_s,
        rss_kb            = startup_rss_kb,
        rss_peak_kb       = max_rss_kb,
        results           = results
    )
    jsonlite::write_json(out, path, auto_unbox = TRUE, digits = 10)
}

# ---- Startup timing -----------------------------------------------------
startup_start <- proc.time()

suppressPackageStartupMessages(library(jsonlite))

startup_end <- proc.time()
startup_s <- as.numeric((startup_end - startup_start)["elapsed"])

# Capture RSS and R heap after startup (before any scenarios)
startup_rss_kb <- read_vmrss()
session_max_rss_kb <- startup_rss_kb
gc_usage <- gc()
# gc() returns matrix: column 1 = raw counts, column 2 = MB equivalent
startup_r_heap_mb <- gc_usage[1, 2] + gc_usage[2, 2]

# ---- Read scenarios ----------------------------------------------------
scenarios <- jsonlite::read_json(scenarios_path, simplifyVector = FALSE)

# ---- Run scenarios -----------------------------------------------------
results <- list()

for (k in seq_along(scenarios)) {
    scenario <- scenarios[[k]]
    sid <- scenario$id %||% sprintf("scenario_%d", k)

    message(sprintf("[%d/%d] %s ...", k, length(scenarios), sid))

    # RSS before
    rss_before_kb <- read_vmrss()

    # R heap before (gc() forces GC, col 2 = MB)
    gc_before <- gc()
    r_heap_mb <- gc_before[1, 2] + gc_before[2, 2]

    # Timer wraps ONLY harmonizR()
    t0 <- proc.time()
    res <- harmonizr_call(scenario)
    t1 <- proc.time()

    # RSS after + delta
    rss_after_kb <- read_vmrss()
    rss_delta_kb <- rss_after_kb - rss_before_kb
    if (rss_after_kb > session_max_rss_kb) {
        session_max_rss_kb <- rss_after_kb
    }

    # R heap after (another GC to measure what the algorithm actually allocated)
    gc_after <- gc()
    r_heap_after_mb <- gc_after[1, 2] + gc_after[2, 2]
    r_heap_delta_mb <- r_heap_after_mb - r_heap_mb

    elapsed <- t1 - t0
    elapsed_s <- as.numeric(elapsed["elapsed"])
    cpu_user_s <- as.numeric(elapsed["user.self"])
    cpu_sys_s  <- as.numeric(elapsed["sys.self"])

    # Write concordance TSV AFTER timer stops (I/O not timed)
    output_tsv <- scenario$output_tsv
    features_out <- NA_integer_
    error_msg <- res$error

    if (is.null(error_msg) && is.data.frame(res)) {
        features_out <- nrow(res)
        dir.create(dirname(output_tsv), showWarnings = FALSE, recursive = TRUE)
        write.table(res, file = output_tsv, sep = "\t", quote = FALSE, col.names = NA)
    }

    entry <- list(
        id              = sid,
        times_s         = elapsed_s,
        cpu_user_s      = cpu_user_s,
        cpu_sys_s       = cpu_sys_s,
        rss_kb          = rss_after_kb,
        rss_delta_kb    = rss_delta_kb,
        r_heap_mb       = r_heap_after_mb,
        r_heap_delta_mb = r_heap_delta_mb,
        features_out    = features_out,
        output_tsv      = if (is.data.frame(res)) output_tsv else NULL,
        error           = error_msg %||% NA_character_
    )
    results[[k]] <- entry

    # Partial JSON write (crash resilience)
    write_partial_json(results, output_path, session_max_rss_kb)

    message(sprintf("  done (%.3fs)", elapsed_s))
}

# ---- Write final output -------------------------------------------------
write_partial_json(results, output_path, session_max_rss_kb)
message(sprintf("Results written to %s", output_path))
