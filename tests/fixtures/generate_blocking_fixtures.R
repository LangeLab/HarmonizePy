#!/usr/bin/env Rscript
# Generate R fixtures for Phase 2: blocking, sorting, and unique-removal.
#
# Run from the project root:
#   Rscript tests/fixtures/generate_blocking_fixtures.R
#
# Requires: HarmonizR >= 1.8.0 (for sort, block, ur parameters).
# Reads existing input TSVs from tests/fixtures/ to guarantee data identity
# with the Python-side tests.

library(HarmonizR)

out_dir <- "tests/fixtures"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

write_matrix <- function(mat, path) {
  df <- data.frame(feature = rownames(mat), mat, check.names = FALSE)
  write.table(df, path, sep = "\t", quote = FALSE, row.names = FALSE)
}

# ---------------------------------------------------------------------------
# Load the three shared input datasets
# ---------------------------------------------------------------------------
cat("Loading input data...\n")

med_raw <- read.table(
  file.path(out_dir, "medium_input.tsv"),
  sep = "\t", header = TRUE, row.names = 1, check.names = FALSE
)
data_med <- as.matrix(med_raw)
desc_med <- read.csv(file.path(out_dir, "medium_batch.csv"), check.names = FALSE)

hm_raw <- read.table(
  file.path(out_dir, "highmiss_input.tsv"),
  sep = "\t", header = TRUE, row.names = 1, check.names = FALSE
)
data_hm <- as.matrix(hm_raw)
desc_hm <- read.csv(file.path(out_dir, "highmiss_batch.csv"), check.names = FALSE)

lg_raw <- read.table(
  file.path(out_dir, "large_input.tsv"),
  sep = "\t", header = TRUE, row.names = 1, check.names = FALSE
)
data_lg <- as.matrix(lg_raw)
desc_lg <- read.csv(file.path(out_dir, "large_batch.csv"), check.names = FALSE)


# =======================================================================
# A. MEDIUM + block=2, sort=FALSE, ur=TRUE — ComBat modes 1-4 + limma
#    medium:  100 features x 12 samples x 3 batches, 30% missing
# =======================================================================
cat("\n--- A: medium + block=2, no sort ---\n")

for (m in 1:4) {
  tryCatch({
    result <- harmonizR(
      data_as_input        = as.data.frame(data_med),
      description_as_input = desc_med,
      algorithm            = "ComBat",
      ComBat_mode          = m,
      sort                 = FALSE,
      block                = 2,
      ur                   = TRUE,
      plot                 = FALSE,
      output_file          = FALSE,
      verbosity            = 0,
      cores                = 1
    )
    write_matrix(
      as.matrix(result),
      file.path(out_dir, sprintf("medium_block2_combat_mode%d.tsv", m))
    )
    cat(sprintf("  ✓ medium block=2 ComBat mode %d\n", m))
  }, error = function(e) {
    cat(sprintf("  ✗ medium block=2 ComBat mode %d: %s\n", m, e$message))
  })
}

tryCatch({
  result <- harmonizR(
    data_as_input        = as.data.frame(data_med),
    description_as_input = desc_med,
    algorithm            = "limma",
    sort                 = FALSE,
    block                = 2,
    ur                   = TRUE,
    plot                 = FALSE,
    output_file          = FALSE,
    verbosity            = 0,
    cores                = 1
  )
  write_matrix(as.matrix(result), file.path(out_dir, "medium_block2_limma.tsv"))
  cat("  ✓ medium block=2 limma\n")
}, error = function(e) {
  cat(sprintf("  ✗ medium block=2 limma: %s\n", e$message))
})


# =======================================================================
# B. HIGHMISS + block=2, sort=FALSE, ur=TRUE — ComBat modes 1-4 + limma
#    highmiss: 50 features x 9 samples x 3 batches, ~65% missing
# =======================================================================
cat("\n--- B: highmiss + block=2, no sort ---\n")

for (m in 1:4) {
  tryCatch({
    result <- harmonizR(
      data_as_input        = as.data.frame(data_hm),
      description_as_input = desc_hm,
      algorithm            = "ComBat",
      ComBat_mode          = m,
      sort                 = FALSE,
      block                = 2,
      ur                   = TRUE,
      plot                 = FALSE,
      output_file          = FALSE,
      verbosity            = 0,
      cores                = 1
    )
    write_matrix(
      as.matrix(result),
      file.path(out_dir, sprintf("highmiss_block2_combat_mode%d.tsv", m))
    )
    cat(sprintf("  ✓ highmiss block=2 ComBat mode %d\n", m))
  }, error = function(e) {
    cat(sprintf("  ✗ highmiss block=2 ComBat mode %d: %s\n", m, e$message))
  })
}

tryCatch({
  result <- harmonizR(
    data_as_input        = as.data.frame(data_hm),
    description_as_input = desc_hm,
    algorithm            = "limma",
    sort                 = FALSE,
    block                = 2,
    ur                   = TRUE,
    plot                 = FALSE,
    output_file          = FALSE,
    verbosity            = 0,
    cores                = 1
  )
  write_matrix(as.matrix(result), file.path(out_dir, "highmiss_block2_limma.tsv"))
  cat("  ✓ highmiss block=2 limma\n")
}, error = function(e) {
  cat(sprintf("  ✗ highmiss block=2 limma: %s\n", e$message))
})


# =======================================================================
# C. LARGE + block=4, sort=FALSE, ur=TRUE — ComBat mode 1
#    large:   500 features x 30 samples x 5 batches, no missing
#    5 batches → block=4 forms groups {1,2,3,4} and {5}
# =======================================================================
cat("\n--- C: large + block=4, no sort, mode 1 ---\n")

tryCatch({
  result <- harmonizR(
    data_as_input        = as.data.frame(data_lg),
    description_as_input = desc_lg,
    algorithm            = "ComBat",
    ComBat_mode          = 1,
    sort                 = FALSE,
    block                = 4,
    ur                   = TRUE,
    plot                 = FALSE,
    output_file          = FALSE,
    verbosity            = 0,
    cores                = 1
  )
  write_matrix(as.matrix(result), file.path(out_dir, "large_block4_combat_mode1.tsv"))
  cat("  ✓ large block=4 ComBat mode 1\n")
}, error = function(e) {
  cat(sprintf("  ✗ large block=4 ComBat mode 1: %s\n", e$message))
})


# =======================================================================
# D. MEDIUM + sort=sparsity_sort + block=2, ur=TRUE
#    ComBat modes 1-4 + limma
# =======================================================================
cat("\n--- D: medium + sparsity_sort + block=2 ---\n")

for (m in 1:4) {
  tryCatch({
    result <- harmonizR(
      data_as_input        = as.data.frame(data_med),
      description_as_input = desc_med,
      algorithm            = "ComBat",
      ComBat_mode          = m,
      sort                 = "sparsity_sort",
      block                = 2,
      ur                   = TRUE,
      plot                 = FALSE,
      output_file          = FALSE,
      verbosity            = 0,
      cores                = 1
    )
    write_matrix(
      as.matrix(result),
      file.path(out_dir, sprintf("medium_sparsity_block2_combat_mode%d.tsv", m))
    )
    cat(sprintf("  ✓ medium sparsity block=2 ComBat mode %d\n", m))
  }, error = function(e) {
    cat(sprintf("  ✗ medium sparsity block=2 ComBat mode %d: %s\n", m, e$message))
  })
}

tryCatch({
  result <- harmonizR(
    data_as_input        = as.data.frame(data_med),
    description_as_input = desc_med,
    algorithm            = "limma",
    sort                 = "sparsity_sort",
    block                = 2,
    ur                   = TRUE,
    plot                 = FALSE,
    output_file          = FALSE,
    verbosity            = 0,
    cores                = 1
  )
  write_matrix(as.matrix(result), file.path(out_dir, "medium_sparsity_block2_limma.tsv"))
  cat("  ✓ medium sparsity block=2 limma\n")
}, error = function(e) {
  cat(sprintf("  ✗ medium sparsity block=2 limma: %s\n", e$message))
})


# =======================================================================
# E. MEDIUM + sort=jaccard_sort + block=2, ur=TRUE — ComBat mode 1
# =======================================================================
cat("\n--- E: medium + jaccard_sort + block=2, mode 1 ---\n")

tryCatch({
  result <- harmonizR(
    data_as_input        = as.data.frame(data_med),
    description_as_input = desc_med,
    algorithm            = "ComBat",
    ComBat_mode          = 1,
    sort                 = "jaccard_sort",
    block                = 2,
    ur                   = TRUE,
    plot                 = FALSE,
    output_file          = FALSE,
    verbosity            = 0,
    cores                = 1
  )
  write_matrix(
    as.matrix(result),
    file.path(out_dir, "medium_jaccard_block2_combat_mode1.tsv")
  )
  cat("  ✓ medium jaccard block=2 ComBat mode 1\n")
}, error = function(e) {
  cat(sprintf("  ✗ medium jaccard block=2 ComBat mode 1: %s\n", e$message))
})


# =======================================================================
# F. MEDIUM + sort=seriation_sort + block=2, ur=TRUE — ComBat mode 1
# =======================================================================
cat("\n--- F: medium + seriation_sort + block=2, mode 1 ---\n")

tryCatch({
  result <- harmonizR(
    data_as_input        = as.data.frame(data_med),
    description_as_input = desc_med,
    algorithm            = "ComBat",
    ComBat_mode          = 1,
    sort                 = "seriation_sort",
    block                = 2,
    ur                   = TRUE,
    plot                 = FALSE,
    output_file          = FALSE,
    verbosity            = 0,
    cores                = 1
  )
  write_matrix(
    as.matrix(result),
    file.path(out_dir, "medium_seriation_block2_combat_mode1.tsv")
  )
  cat("  ✓ medium seriation block=2 ComBat mode 1\n")
}, error = function(e) {
  cat(sprintf("  ✗ medium seriation block=2 ComBat mode 1: %s\n", e$message))
})


# =======================================================================
# G. LARGE + sort=sparsity_sort + block=4, ur=TRUE — ComBat mode 1
#    large: 5 batches → block=4 is valid (4 < 5)
# =======================================================================
cat("\n--- G: large + sparsity_sort + block=4, mode 1 ---\n")

tryCatch({
  result <- harmonizR(
    data_as_input        = as.data.frame(data_lg),
    description_as_input = desc_lg,
    algorithm            = "ComBat",
    ComBat_mode          = 1,
    sort                 = "sparsity_sort",
    block                = 4,
    ur                   = TRUE,
    plot                 = FALSE,
    output_file          = FALSE,
    verbosity            = 0,
    cores                = 1
  )
  write_matrix(
    as.matrix(result),
    file.path(out_dir, "large_sparsity_block4_combat_mode1.tsv")
  )
  cat("  ✓ large sparsity block=4 ComBat mode 1\n")
}, error = function(e) {
  cat(sprintf("  ✗ large sparsity block=4 ComBat mode 1: %s\n", e$message))
})


# =======================================================================
# H. HIGHMISS + sort=sparsity_sort + block=2, ur=TRUE — ComBat mode 1
# =======================================================================
cat("\n--- H: highmiss + sparsity_sort + block=2, mode 1 ---\n")

tryCatch({
  result <- harmonizR(
    data_as_input        = as.data.frame(data_hm),
    description_as_input = desc_hm,
    algorithm            = "ComBat",
    ComBat_mode          = 1,
    sort                 = "sparsity_sort",
    block                = 2,
    ur                   = TRUE,
    plot                 = FALSE,
    output_file          = FALSE,
    verbosity            = 0,
    cores                = 1
  )
  write_matrix(
    as.matrix(result),
    file.path(out_dir, "highmiss_sparsity_block2_combat_mode1.tsv")
  )
  cat("  ✓ highmiss sparsity block=2 ComBat mode 1\n")
}, error = function(e) {
  cat(sprintf("  ✗ highmiss sparsity block=2 ComBat mode 1: %s\n", e$message))
})


# =======================================================================
# I. HIGHMISS + ur=FALSE, sort=FALSE, block=NULL — ComBat mode 1
#    Baseline for unique-removal comparison (ur=TRUE already exists as
#    highmiss_harmonizr_combat_mode1.tsv from generate_edgecase_fixtures.R)
# =======================================================================
cat("\n--- I: highmiss + ur=FALSE, no block, mode 1 ---\n")

tryCatch({
  result <- harmonizR(
    data_as_input        = as.data.frame(data_hm),
    description_as_input = desc_hm,
    algorithm            = "ComBat",
    ComBat_mode          = 1,
    sort                 = FALSE,
    block                = NULL,
    ur                   = FALSE,
    plot                 = FALSE,
    output_file          = FALSE,
    verbosity            = 0,
    cores                = 1
  )
  write_matrix(as.matrix(result), file.path(out_dir, "highmiss_nour_combat_mode1.tsv"))
  cat("  ✓ highmiss ur=FALSE ComBat mode 1\n")
}, error = function(e) {
  cat(sprintf("  ✗ highmiss ur=FALSE ComBat mode 1: %s\n", e$message))
})


cat("\n=== Blocking fixture generation complete ===\n")
