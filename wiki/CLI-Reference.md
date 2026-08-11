# CLI Reference

HarmonizePy includes a command-line interface that exposes the full `harmonize()` pipeline.

```bash
harmonizepy data.tsv batch.csv [options]
```

---

## Basic examples

```bash
# Minimal: ComBat mode 1, default output path
harmonizepy data.tsv batch.csv

# Explicit output path
harmonizepy data.tsv batch.csv -o corrected.tsv

# limma, write parquet
harmonizepy data.tsv batch.csv --algorithm limma -o corrected.parquet

# Sort by sparsity and block into pairs
harmonizepy data.tsv batch.csv --sort sparsity --block 2 -o corrected.tsv

# Validate inputs and print run plan without computing
harmonizepy data.tsv batch.csv --dry-run

# Save a reproducible JSON summary alongside the result
harmonizepy data.tsv batch.csv -o corrected.tsv --summary run.json
```

---

## Positional arguments

- `data`: features x samples matrix (TSV, CSV, or Parquet). The first column is treated as feature names.
- `description`: batch description CSV with columns `ID`, `sample`, `batch`.

---

## Output flags

- `-o`, `--output PATH` (default: `<data_stem>_corrected.tsv`): output file path. Format is inferred from the extension (`.tsv`, `.csv`, `.parquet`, `.pq`).
- `--output-format {tsv,csv,parquet}` (default: inferred from extension): force a specific output format regardless of file extension. Parquet requires `harmonizepy[io]`.

When `-o` is omitted, the output is written next to the input data file as `<data_stem>_corrected.tsv`. If `--output-format parquet` is given without `-o`, the default becomes `<data_stem>_corrected.parquet`.

---

## Algorithm flags

| Flag | Default | Description |
| --- | --- | --- |
| `--algorithm {ComBat,limma}` | `ComBat` | Batch correction algorithm. |
| `--combat-mode {1,2,3,4}` | `1` | ComBat variant. Ignored when `--algorithm limma`. |
| `--needed-values N` | auto | Minimum non-missing values per batch to qualify a feature. Auto: 2 for modes 1, 3 and limma; 1 for modes 2, 4. |

---

## Pipeline flags

| Flag | Default | Description |
| --- | --- | --- |
| `--sort {sparsity,jaccard,seriation}` | none | Sort batches before blocking so similar batches are grouped together. |
| `--block N` | none | Group N consecutive sorted batches into one sub-matrix block. Must be >= 2 and < total number of batches. |
| `--unique-removal` / `--no-unique-removal` | enabled | Rescue singleton-affiliation features. Disable with `--no-unique-removal` for strict missingness-pattern matching. |

---

## Workflow helpers

- `--dry-run`: validate inputs, print run plan, exit without computing. Exit code `0` if valid, `1` on error.
- `--summary PATH`: write a JSON run summary to PATH after completion. Contains resolved parameters, input/output dimensions, and version.
- `--json`: print the run summary as JSON to stdout. Suppresses INFO log messages.
- `--config PATH`: load flags from a TOML, JSON, or YAML config file. CLI flags override config file values.

---

## Verbosity flags

- `-v`, `--verbose`: enable debug logging (sub-matrix dimensions, affiliation counts, timing).
- `-q`, `--quiet`: suppress all progress messages. Only warnings and errors are shown.
- `--log-file PATH`: write a detailed execution log to PATH (default: `<output_stem>.log`).
- `--no-log`: disable file logging entirely.

---

## Config files

Pass `--config path/to/config.toml` to load settings from a file. TOML, JSON, and YAML are supported. YAML requires the `harmonizepy[config]` extra.

### TOML example

```toml
algorithm = "ComBat"
combat_mode = 1
sort = "sparsity"
block = 2
unique_removal = true
output = "corrected.tsv"
```

### JSON example

```json
{
  "algorithm": "ComBat",
  "combat_mode": 1,
  "sort": "sparsity",
  "block": 2,
  "unique_removal": true
}
```

Config file keys map 1:1 to CLI flag names: `algorithm`, `combat_mode`, `needed_values`, `sort`, `block`, `unique_removal`, `output`, `output_format`, `summary`. CLI flags override config file values. Config file values override built-in defaults.

Config paths are resolved relative to the config file, not the current working directory.

---

## Dry-run output

`--dry-run` validates inputs and prints a summary without running correction:

```bash
HarmonizePy 0.3.2 dry run
────────────────────────────────────────────────────
Features:        1500
Samples:         45
Batches:         5
Sub-matrices:    3  (unique affiliation groups)
Algorithm:        ComBat mode 1
Sort strategy:    sparsity
Block size:       2
Unique removal:   enabled
Inputs valid. Use without --dry-run to run correction.
```

---

## Exit codes

Exit code `0` on success (or `--dry-run` with valid inputs). Exit code `1` on validation error or unrecoverable runtime error.
