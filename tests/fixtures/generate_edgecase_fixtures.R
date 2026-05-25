#!/usr/bin/env Rscript
# Generate comprehensive R edge-case fixtures for HarmonizePy validation.
#
# Run from the project root:
#   Rscript tests/fixtures/generate_edgecase_fixtures.R
#
# These complement the main fixtures (generate_r_fixtures.R) with stress
# tests covering boundary conditions, numerical edge cases, many-batch
# scenarios, unbalanced designs, etc.

library(sva)
library(limma)
library(HarmonizR)

out_dir <- "tests/fixtures"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

write_matrix <- function(mat, path) {
  df <- data.frame(feature = rownames(mat), mat, check.names = FALSE)
  write.table(df, path, sep = "\t", quote = FALSE, row.names = FALSE)
}

# =======================================================================
# 1. LARGE: 500 proteins × 30 samples × 5 batches, no missing
# =======================================================================
cat("--- large (500×30, 5 batches) ---\n")
set.seed(101)
np <- 500; ns <- 30
data_large <- matrix(rnorm(np * ns, mean = 10, sd = 3), nrow = np)
batch_large <- rep(1:5, each = 6)
for (b in 1:5) {
  cols <- which(batch_large == b)
  data_large[, cols] <- data_large[, cols] + rnorm(1, sd = 2)
  data_large[, cols] <- data_large[, cols] * (1 + rnorm(1, sd = 0.3))
}
rownames(data_large) <- paste0("protein_", seq_len(np))
colnames(data_large) <- paste0("sample_", seq_len(ns))
write_matrix(data_large, file.path(out_dir, "large_input.tsv"))
desc_large <- data.frame(ID = colnames(data_large), sample = seq_len(ns), batch = batch_large)
write.csv(desc_large, file.path(out_dir, "large_batch.csv"), row.names = FALSE)

for (m in 1:4) {
  modes <- list(
    list(par.prior = TRUE, mean.only = FALSE),
    list(par.prior = TRUE, mean.only = TRUE),
    list(par.prior = FALSE, mean.only = FALSE),
    list(par.prior = FALSE, mean.only = TRUE)
  )
  tryCatch({
    result <- ComBat(dat = data_large, batch = batch_large,
                     par.prior = modes[[m]]$par.prior,
                     mean.only = modes[[m]]$mean.only)
    write_matrix(result, file.path(out_dir, sprintf("large_combat_mode%d.tsv", m)))
    cat(sprintf("✓ large ComBat mode %d\n", m))
  }, error = function(e) cat(sprintf("✗ large ComBat mode %d: %s\n", m, e$message)))
}
tryCatch({
  result <- removeBatchEffect(data_large, batch = batch_large)
  write_matrix(result, file.path(out_dir, "large_limma.tsv"))
  cat("✓ large limma\n")
}, error = function(e) cat(sprintf("✗ large limma: %s\n", e$message)))

# =======================================================================
# 2. UNBALANCED: 50 proteins × 15 samples × 3 batches (3, 5, 7 samples)
# =======================================================================
cat("\n--- unbalanced (50×15, batches 3/5/7) ---\n")
set.seed(202)
np2 <- 50; ns2 <- 15
data_unbal <- matrix(rnorm(np2 * ns2, mean = 10, sd = 2), nrow = np2)
batch_unbal <- c(rep(1, 3), rep(2, 5), rep(3, 7))
data_unbal[, batch_unbal == 2] <- data_unbal[, batch_unbal == 2] + 1.5
data_unbal[, batch_unbal == 3] <- data_unbal[, batch_unbal == 3] - 2.0
rownames(data_unbal) <- paste0("protein_", seq_len(np2))
colnames(data_unbal) <- paste0("sample_", seq_len(ns2))
write_matrix(data_unbal, file.path(out_dir, "unbalanced_input.tsv"))
desc_unbal <- data.frame(ID = colnames(data_unbal), sample = seq_len(ns2), batch = batch_unbal)
write.csv(desc_unbal, file.path(out_dir, "unbalanced_batch.csv"), row.names = FALSE)

for (m in 1:4) {
  modes <- list(
    list(par.prior = TRUE, mean.only = FALSE),
    list(par.prior = TRUE, mean.only = TRUE),
    list(par.prior = FALSE, mean.only = FALSE),
    list(par.prior = FALSE, mean.only = TRUE)
  )
  tryCatch({
    result <- ComBat(dat = data_unbal, batch = batch_unbal,
                     par.prior = modes[[m]]$par.prior,
                     mean.only = modes[[m]]$mean.only)
    write_matrix(result, file.path(out_dir, sprintf("unbalanced_combat_mode%d.tsv", m)))
    cat(sprintf("✓ unbalanced ComBat mode %d\n", m))
  }, error = function(e) cat(sprintf("✗ unbalanced ComBat mode %d: %s\n", m, e$message)))
}
tryCatch({
  result <- removeBatchEffect(data_unbal, batch = batch_unbal)
  write_matrix(result, file.path(out_dir, "unbalanced_limma.tsv"))
  cat("✓ unbalanced limma\n")
}, error = function(e) cat(sprintf("✗ unbalanced limma: %s\n", e$message)))

# =======================================================================
# 3. MINIMAL: 2 proteins × 4 samples × 2 batches (smallest valid case)
# =======================================================================
cat("\n--- minimal (2×4, 2 batches) ---\n")
set.seed(303)
data_min <- matrix(c(10, 12, 8, 14, 15, 17, 13, 19), nrow = 2, byrow = TRUE)
batch_min <- c(1, 1, 2, 2)
rownames(data_min) <- c("protein_1", "protein_2")
colnames(data_min) <- paste0("sample_", 1:4)
write_matrix(data_min, file.path(out_dir, "minimal_input.tsv"))
desc_min <- data.frame(ID = colnames(data_min), sample = 1:4, batch = batch_min)
write.csv(desc_min, file.path(out_dir, "minimal_batch.csv"), row.names = FALSE)

for (m in 1:4) {
  modes <- list(
    list(par.prior = TRUE, mean.only = FALSE),
    list(par.prior = TRUE, mean.only = TRUE),
    list(par.prior = FALSE, mean.only = FALSE),
    list(par.prior = FALSE, mean.only = TRUE)
  )
  tryCatch({
    result <- ComBat(dat = data_min, batch = batch_min,
                     par.prior = modes[[m]]$par.prior,
                     mean.only = modes[[m]]$mean.only)
    write_matrix(result, file.path(out_dir, sprintf("minimal_combat_mode%d.tsv", m)))
    cat(sprintf("✓ minimal ComBat mode %d\n", m))
  }, error = function(e) cat(sprintf("✗ minimal ComBat mode %d: %s\n", m, e$message)))
}
tryCatch({
  result <- removeBatchEffect(data_min, batch = batch_min)
  write_matrix(result, file.path(out_dir, "minimal_limma.tsv"))
  cat("✓ minimal limma\n")
}, error = function(e) cat(sprintf("✗ minimal limma: %s\n", e$message)))

# =======================================================================
# 4. HIGH VARIANCE: extreme batch effects (shift=20, scale=5)
# =======================================================================
cat("\n--- highvar (30×8, extreme effects) ---\n")
set.seed(404)
np4 <- 30; ns4 <- 8
data_hv <- matrix(rnorm(np4 * ns4, mean = 10, sd = 1), nrow = np4)
batch_hv <- rep(1:2, each = 4)
data_hv[, batch_hv == 2] <- data_hv[, batch_hv == 2] * 5 + 20
rownames(data_hv) <- paste0("protein_", seq_len(np4))
colnames(data_hv) <- paste0("sample_", seq_len(ns4))
write_matrix(data_hv, file.path(out_dir, "highvar_input.tsv"))
desc_hv <- data.frame(ID = colnames(data_hv), sample = seq_len(ns4), batch = batch_hv)
write.csv(desc_hv, file.path(out_dir, "highvar_batch.csv"), row.names = FALSE)

for (m in 1:4) {
  modes <- list(
    list(par.prior = TRUE, mean.only = FALSE),
    list(par.prior = TRUE, mean.only = TRUE),
    list(par.prior = FALSE, mean.only = FALSE),
    list(par.prior = FALSE, mean.only = TRUE)
  )
  tryCatch({
    result <- ComBat(dat = data_hv, batch = batch_hv,
                     par.prior = modes[[m]]$par.prior,
                     mean.only = modes[[m]]$mean.only)
    write_matrix(result, file.path(out_dir, sprintf("highvar_combat_mode%d.tsv", m)))
    cat(sprintf("✓ highvar ComBat mode %d\n", m))
  }, error = function(e) cat(sprintf("✗ highvar ComBat mode %d: %s\n", m, e$message)))
}
tryCatch({
  result <- removeBatchEffect(data_hv, batch = batch_hv)
  write_matrix(result, file.path(out_dir, "highvar_limma.tsv"))
  cat("✓ highvar limma\n")
}, error = function(e) cat(sprintf("✗ highvar limma: %s\n", e$message)))

# =======================================================================
# 5. NEAR-CONSTANT: some features with very low variance  
# =======================================================================
cat("\n--- nearconstant (20×6, near-zero variance features) ---\n")
set.seed(505)
np5 <- 20; ns5 <- 6
data_nc <- matrix(rnorm(np5 * ns5, mean = 10, sd = 2), nrow = np5)
# Make first 3 features near-constant (variance ~ 1e-6)
data_nc[1, ] <- 10.0 + rnorm(ns5, sd = 1e-3)
data_nc[2, ] <- 5.0 + rnorm(ns5, sd = 1e-3)
data_nc[3, ] <- 15.0 + rnorm(ns5, sd = 1e-3)
batch_nc <- rep(1:2, each = 3)
data_nc[, batch_nc == 2] <- data_nc[, batch_nc == 2] + 1.5
rownames(data_nc) <- paste0("protein_", seq_len(np5))
colnames(data_nc) <- paste0("sample_", seq_len(ns5))
write_matrix(data_nc, file.path(out_dir, "nearconstant_input.tsv"))
desc_nc <- data.frame(ID = colnames(data_nc), sample = seq_len(ns5), batch = batch_nc)
write.csv(desc_nc, file.path(out_dir, "nearconstant_batch.csv"), row.names = FALSE)

for (m in c(2, 4)) {
  # Mode 2 and 4 (mean_only) should handle near-constant features fine
  modes <- list(`2` = list(par.prior = TRUE, mean.only = TRUE),
                `4` = list(par.prior = FALSE, mean.only = TRUE))
  tryCatch({
    result <- ComBat(dat = data_nc, batch = batch_nc,
                     par.prior = modes[[as.character(m)]]$par.prior,
                     mean.only = modes[[as.character(m)]]$mean.only)
    write_matrix(result, file.path(out_dir, sprintf("nearconstant_combat_mode%d.tsv", m)))
    cat(sprintf("✓ nearconstant ComBat mode %d\n", m))
  }, error = function(e) cat(sprintf("✗ nearconstant ComBat mode %d: %s\n", m, e$message)))
}
tryCatch({
  result <- removeBatchEffect(data_nc, batch = batch_nc)
  write_matrix(result, file.path(out_dir, "nearconstant_limma.tsv"))
  cat("✓ nearconstant limma\n")
}, error = function(e) cat(sprintf("✗ nearconstant limma: %s\n", e$message)))

# =======================================================================
# 6. NEGATIVE VALUES: data with negative values (log2FC or centered data)
# =======================================================================
cat("\n--- negative (40×8, centered data with negatives) ---\n")
set.seed(606)
np6 <- 40; ns6 <- 8
data_neg <- matrix(rnorm(np6 * ns6, mean = 0, sd = 3), nrow = np6)
batch_neg <- rep(1:2, each = 4)
data_neg[, batch_neg == 2] <- data_neg[, batch_neg == 2] + 4.0
rownames(data_neg) <- paste0("protein_", seq_len(np6))
colnames(data_neg) <- paste0("sample_", seq_len(ns6))
write_matrix(data_neg, file.path(out_dir, "negative_input.tsv"))
desc_neg <- data.frame(ID = colnames(data_neg), sample = seq_len(ns6), batch = batch_neg)
write.csv(desc_neg, file.path(out_dir, "negative_batch.csv"), row.names = FALSE)

for (m in 1:4) {
  modes <- list(
    list(par.prior = TRUE, mean.only = FALSE),
    list(par.prior = TRUE, mean.only = TRUE),
    list(par.prior = FALSE, mean.only = FALSE),
    list(par.prior = FALSE, mean.only = TRUE)
  )
  tryCatch({
    result <- ComBat(dat = data_neg, batch = batch_neg,
                     par.prior = modes[[m]]$par.prior,
                     mean.only = modes[[m]]$mean.only)
    write_matrix(result, file.path(out_dir, sprintf("negative_combat_mode%d.tsv", m)))
    cat(sprintf("✓ negative ComBat mode %d\n", m))
  }, error = function(e) cat(sprintf("✗ negative ComBat mode %d: %s\n", m, e$message)))
}
tryCatch({
  result <- removeBatchEffect(data_neg, batch = batch_neg)
  write_matrix(result, file.path(out_dir, "negative_limma.tsv"))
  cat("✓ negative limma\n")
}, error = function(e) cat(sprintf("✗ negative limma: %s\n", e$message)))

# =======================================================================
# 7. SINGLETON: one batch has only 1 sample (forces mean_only in R)
# =======================================================================
cat("\n--- singleton (20×5, one batch has 1 sample) ---\n")
set.seed(707)
np7 <- 20; ns7 <- 5
data_sing <- matrix(rnorm(np7 * ns7, mean = 10, sd = 2), nrow = np7)
batch_sing <- c(1, 1, 1, 1, 2)
data_sing[, 5] <- data_sing[, 5] + 3.0
rownames(data_sing) <- paste0("protein_", seq_len(np7))
colnames(data_sing) <- paste0("sample_", seq_len(ns7))
write_matrix(data_sing, file.path(out_dir, "singleton_input.tsv"))
desc_sing <- data.frame(ID = colnames(data_sing), sample = seq_len(ns7), batch = batch_sing)
write.csv(desc_sing, file.path(out_dir, "singleton_batch.csv"), row.names = FALSE)

# R sva forces mean_only=TRUE when any batch has 1 sample
for (m in 1:4) {
  modes <- list(
    list(par.prior = TRUE, mean.only = FALSE),
    list(par.prior = TRUE, mean.only = TRUE),
    list(par.prior = FALSE, mean.only = FALSE),
    list(par.prior = FALSE, mean.only = TRUE)
  )
  tryCatch({
    result <- ComBat(dat = data_sing, batch = batch_sing,
                     par.prior = modes[[m]]$par.prior,
                     mean.only = modes[[m]]$mean.only)
    write_matrix(result, file.path(out_dir, sprintf("singleton_combat_mode%d.tsv", m)))
    cat(sprintf("✓ singleton ComBat mode %d\n", m))
  }, error = function(e) cat(sprintf("✗ singleton ComBat mode %d: %s\n", m, e$message)))
}

# =======================================================================
# 8. WIDE: more samples than features (5 features × 20 samples)
# =======================================================================
cat("\n--- wide (5×20, more samples than features) ---\n")
set.seed(808)
np8 <- 5; ns8 <- 20
data_wide <- matrix(rnorm(np8 * ns8, mean = 10, sd = 2), nrow = np8)
batch_wide <- rep(1:4, each = 5)
for (b in 1:4) data_wide[, batch_wide == b] <- data_wide[, batch_wide == b] + rnorm(1, sd = 2)
rownames(data_wide) <- paste0("protein_", seq_len(np8))
colnames(data_wide) <- paste0("sample_", seq_len(ns8))
write_matrix(data_wide, file.path(out_dir, "wide_input.tsv"))
desc_wide <- data.frame(ID = colnames(data_wide), sample = seq_len(ns8), batch = batch_wide)
write.csv(desc_wide, file.path(out_dir, "wide_batch.csv"), row.names = FALSE)

for (m in 1:4) {
  modes <- list(
    list(par.prior = TRUE, mean.only = FALSE),
    list(par.prior = TRUE, mean.only = TRUE),
    list(par.prior = FALSE, mean.only = FALSE),
    list(par.prior = FALSE, mean.only = TRUE)
  )
  tryCatch({
    result <- ComBat(dat = data_wide, batch = batch_wide,
                     par.prior = modes[[m]]$par.prior,
                     mean.only = modes[[m]]$mean.only)
    write_matrix(result, file.path(out_dir, sprintf("wide_combat_mode%d.tsv", m)))
    cat(sprintf("✓ wide ComBat mode %d\n", m))
  }, error = function(e) cat(sprintf("✗ wide ComBat mode %d: %s\n", m, e$message)))
}
tryCatch({
  result <- removeBatchEffect(data_wide, batch = batch_wide)
  write_matrix(result, file.path(out_dir, "wide_limma.tsv"))
  cat("✓ wide limma\n")
}, error = function(e) cat(sprintf("✗ wide limma: %s\n", e$message)))

# =======================================================================
# 9. MEDIUM LIMMA: medium dataset with missing data through HarmonizR limma
# =======================================================================
cat("\n--- medium limma (100×12, 3 batches, 30% missing, limma) ---\n")
# Load the medium dataset from disk (generated by generate_r_fixtures.R) to
# guarantee data identity with the Python-side tests and avoid duplication.
med_raw <- read.table(
  file.path(out_dir, "medium_input.tsv"),
  sep = "\t", header = TRUE, row.names = 1, check.names = FALSE
)
data_med <- as.matrix(med_raw)
desc_med <- read.csv(file.path(out_dir, "medium_batch.csv"), check.names = FALSE)

tryCatch({
  result <- harmonizR(
    data_as_input = as.data.frame(data_med),
    description_as_input = desc_med,
    algorithm = "limma",
    sort = FALSE, plot = FALSE,
    output_file = FALSE,
    verbosity = 0, cores = 1
  )
  write_matrix(as.matrix(result), file.path(out_dir, "medium_harmonizr_limma.tsv"))
  cat("✓ medium HarmonizR limma\n")
}, error = function(e) cat(sprintf("✗ medium HarmonizR limma: %s\n", e$message)))

# =======================================================================
# 10. SPARSE MISSING: 50% of features missing one full batch
# =======================================================================
cat("\n--- sparse (80×9, 3 batches, 50% have one batch missing) ---\n")
set.seed(909)
np10 <- 80; ns10 <- 9
data_sp <- matrix(rnorm(np10 * ns10, mean = 10, sd = 2), nrow = np10)
batch_sp <- rep(1:3, each = 3)
data_sp[, batch_sp == 2] <- data_sp[, batch_sp == 2] + 2.0
data_sp[, batch_sp == 3] <- data_sp[, batch_sp == 3] - 1.5
# Drop one batch for 50% of features
for (i in seq_len(np10)) {
  if (runif(1) < 0.5) {
    drop_batch <- sample(1:3, 1)
    data_sp[i, batch_sp == drop_batch] <- NA
  }
}
rownames(data_sp) <- paste0("protein_", seq_len(np10))
colnames(data_sp) <- paste0("sample_", seq_len(ns10))
write_matrix(data_sp, file.path(out_dir, "sparse_input.tsv"))
desc_sp <- data.frame(ID = colnames(data_sp), sample = seq_len(ns10), batch = batch_sp)
write.csv(desc_sp, file.path(out_dir, "sparse_batch.csv"), row.names = FALSE)

for (m in 1:4) {
  tryCatch({
    result <- harmonizR(
      data_as_input = as.data.frame(data_sp),
      description_as_input = desc_sp,
      algorithm = "ComBat", ComBat_mode = m,
      sort = FALSE, plot = FALSE,
      output_file = file.path(out_dir, sprintf("sparse_harmonizr_combat_mode%d", m)),
      verbosity = 0, cores = 1
    )
    write_matrix(as.matrix(result), file.path(out_dir, sprintf("sparse_harmonizr_combat_mode%d.tsv", m)))
    cat(sprintf("✓ sparse HarmonizR ComBat mode %d\n", m))
  }, error = function(e) cat(sprintf("✗ sparse HarmonizR ComBat mode %d: %s\n", m, e$message)))
}

# =======================================================================
# 11. HIGH-MISSINGNESS: 50 proteins x 9 samples x 3 batches (3 each),
#     ~65% of features have one whole batch missing
# =======================================================================
cat("\n--- highmiss (50x9, 3 batches of 3, 65% structural missing) ---\n")
set.seed(777)
np11 <- 50; ns11 <- 9
data_hm <- matrix(rnorm(np11 * ns11, mean = 10, sd = 2), nrow = np11)
batch_hm <- rep(1:3, each = 3)
data_hm[, batch_hm == 2] <- data_hm[, batch_hm == 2] + 2.5
data_hm[, batch_hm == 3] <- data_hm[, batch_hm == 3] - 1.8
# 65% of features have one whole batch absent
for (i in seq_len(np11)) {
  if (runif(1) < 0.65) {
    drop_batch <- sample(1:3, 1)
    data_hm[i, batch_hm == drop_batch] <- NA
  }
}
rownames(data_hm) <- paste0("protein_", seq_len(np11))
colnames(data_hm) <- paste0("sample_", seq_len(ns11))
write_matrix(data_hm, file.path(out_dir, "highmiss_input.tsv"))
desc_hm <- data.frame(
  ID     = colnames(data_hm),
  sample = seq_len(ns11),
  batch  = batch_hm
)
write.csv(desc_hm, file.path(out_dir, "highmiss_batch.csv"), row.names = FALSE)

for (m in 1:4) {
  tryCatch({
    result <- harmonizR(
      data_as_input        = as.data.frame(data_hm),
      description_as_input = desc_hm,
      algorithm            = "ComBat", ComBat_mode = m,
      sort = FALSE, plot = FALSE,
      output_file          = file.path(out_dir, sprintf("highmiss_harmonizr_combat_mode%d", m)),
      verbosity = 0, cores = 1
    )
    write_matrix(as.matrix(result),
                 file.path(out_dir, sprintf("highmiss_harmonizr_combat_mode%d.tsv", m)))
    cat(sprintf("✓ highmiss HarmonizR ComBat mode %d\n", m))
  }, error = function(e) cat(sprintf("✗ highmiss HarmonizR ComBat mode %d: %s\n", m, e$message)))
}

tryCatch({
  result <- harmonizR(
    data_as_input        = as.data.frame(data_hm),
    description_as_input = desc_hm,
    algorithm            = "limma",
    sort = FALSE, plot = FALSE,
    output_file          = file.path(out_dir, "highmiss_harmonizr_limma"),
    verbosity = 0, cores = 1
  )
  write_matrix(as.matrix(result), file.path(out_dir, "highmiss_harmonizr_limma.tsv"))
  cat("✓ highmiss HarmonizR limma\n")
}, error = function(e) cat(sprintf("✗ highmiss HarmonizR limma: %s\n", e$message)))

cat("\n=== Edge-case fixture generation complete ===\n")
