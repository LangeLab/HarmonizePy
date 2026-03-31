#!/usr/bin/env Rscript
# Generate R concordance fixtures for HarmonizePy.
#
# Produces reference outputs from:
#   1. sva::ComBat (all 4 modes) — for engine-level concordance with
#      harmonizepy.combat.combat()
#   2. HarmonizR::harmonizR (ComBat + limma) — for full-pipeline concordance
#
# Run from the project root:
#   Rscript tests/fixtures/generate_r_fixtures.R
#
# Outputs land in tests/fixtures/

library(sva)
library(HarmonizR)

# Determine output directory: prefer tests/fixtures relative to working dir
out_dir <- "tests/fixtures"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

cat("Output directory:", normalizePath(out_dir), "\n")

# ==========================================================================
# Helper: write a matrix as TSV (row names in first column)
# ==========================================================================
write_matrix <- function(mat, path) {
  df <- data.frame(feature = rownames(mat), mat, check.names = FALSE)
  write.table(df, path, sep = "\t", quote = FALSE, row.names = FALSE)
}

# ==========================================================================
# 1. SMALL TEST CASE — 10 proteins × 6 samples × 2 batches, no missing
# ==========================================================================
set.seed(42)
n_prot <- 10
n_samp <- 6

data_small <- matrix(rnorm(n_prot * n_samp, mean = 10, sd = 2), nrow = n_prot)
# Add batch effect to samples 4-6
data_small[, 4:6] <- data_small[, 4:6] + 2.5

rownames(data_small) <- paste0("protein_", seq_len(n_prot))
colnames(data_small) <- paste0("sample_", seq_len(n_samp))

batch_small <- rep(1:2, each = 3)

# Save input
write_matrix(data_small, file.path(out_dir, "small_input.tsv"))

batch_df_small <- data.frame(
  ID = colnames(data_small),
  sample = seq_len(n_samp),
  batch = batch_small
)
write.csv(batch_df_small, file.path(out_dir, "small_batch.csv"),
          row.names = FALSE)

# --- sva::ComBat direct (all 4 modes) ---
modes <- list(
  list(par.prior = TRUE,  mean.only = FALSE),
  list(par.prior = TRUE,  mean.only = TRUE),
  list(par.prior = FALSE, mean.only = FALSE),
  list(par.prior = FALSE, mean.only = TRUE)
)

for (m in seq_along(modes)) {
  tryCatch({
    result <- ComBat(
      dat        = data_small,
      batch      = batch_small,
      par.prior  = modes[[m]]$par.prior,
      mean.only  = modes[[m]]$mean.only
    )
    write_matrix(
      result,
      file.path(out_dir, sprintf("small_combat_mode%d.tsv", m))
    )
    cat(sprintf("✓ small ComBat mode %d\n", m))
  }, error = function(e) {
    cat(sprintf("✗ small ComBat mode %d: %s\n", m, e$message))
  })
}

# --- limma direct ---
library(limma)
tryCatch({
  result <- removeBatchEffect(data_small, batch = batch_small)
  write_matrix(result, file.path(out_dir, "small_limma.tsv"))
  cat("OK small limma\n")
}, error = function(e) {
  cat(sprintf("FAIL small limma: %s\n", e$message))
})

# ==========================================================================
# 2. MEDIUM TEST CASE — 100 proteins × 12 samples × 3 batches, 30% missing
# ==========================================================================
set.seed(123)
n_prot2 <- 100
n_samp2 <- 12

data_med <- matrix(rnorm(n_prot2 * n_samp2, mean = 10, sd = 2), nrow = n_prot2)
data_med[, 5:8]  <- data_med[, 5:8]  + 1.8
data_med[, 9:12] <- data_med[, 9:12] - 1.2

# Inject 30% structural missingness (entire batch missing for some proteins)
missing_mask <- matrix(FALSE, nrow = n_prot2, ncol = n_samp2)
for (i in seq_len(n_prot2)) {
  if (runif(1) < 0.3) {
    drop_batch  <- sample(1:3, 1)
    batch_cols  <- ((drop_batch - 1) * 4 + 1):(drop_batch * 4)
    missing_mask[i, batch_cols] <- TRUE
  }
}
data_med[missing_mask] <- NA

rownames(data_med) <- paste0("protein_", seq_len(n_prot2))
colnames(data_med) <- paste0("sample_", seq_len(n_samp2))

batch_med <- rep(1:3, each = 4)

write_matrix(data_med, file.path(out_dir, "medium_input.tsv"))

batch_df_med <- data.frame(
  ID = colnames(data_med),
  sample = seq_len(n_samp2),
  batch = batch_med
)
write.csv(batch_df_med, file.path(out_dir, "medium_batch.csv"),
          row.names = FALSE)

# --- HarmonizR full pipeline (ComBat, all 4 modes, with missingness) ---
desc_med <- data.frame(
  ID = colnames(data_med),
  sample = seq_len(n_samp2),
  batch = batch_med
)

for (m in 1:4) {
  tryCatch({
    result <- harmonizR(
      data_as_input        = as.data.frame(data_med),
      description_as_input = desc_med,
      algorithm            = "ComBat",
      ComBat_mode          = m,
      sort                 = FALSE,
      plot                 = FALSE,
      output_file          = file.path(out_dir, sprintf("medium_harmonizr_combat_mode%d", m)),
      verbosity            = 0
    )
    write_matrix(
      as.matrix(result),
      file.path(out_dir, sprintf("medium_harmonizr_combat_mode%d.tsv", m))
    )
    cat(sprintf("✓ medium HarmonizR ComBat mode %d\n", m))
  }, error = function(e) {
    cat(sprintf("✗ medium HarmonizR ComBat mode %d: %s\n", m, e$message))
  })
}

cat("\n=== Fixture generation complete ===\n")
cat("Files written to:", normalizePath(out_dir), "\n")
