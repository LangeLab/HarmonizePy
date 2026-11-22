# CLI reference

The whole command line on one page: the arguments, every flag, config files, the dry run, logging and exit statuses. `harmonizepy` runs the same pipeline as `harmonize()`; `python -m harmonizepy` is the same command.

```text
usage: harmonizepy [-h] [-o PATH] [--output-format {tsv,csv,parquet}]
                   [--algorithm {ComBat,limma}] [--combat-mode {1,2,3,4}]
                   [--needed-values N] [--sort {sparsity,jaccard,seriation}]
                   [--block N] [--unique-removal | --no-unique-removal]
                   [--summary PATH] [--report PATH] [--dry-run]
                   [--config PATH] [--json] [-v | -q] [--log-file PATH]
                   [--no-log] [--version]
                   data description
```

## Arguments

- `data`: the data matrix, read by extension: `.csv` comma-separated, `.parquet` or `.pq` Parquet, otherwise tab-separated ([Input format](Input-Format)).
- `description`: the batch description, always read as comma-separated.

## Flags

| Flag                                      | Default                     | Explained on                                 |
| ----------------------------------------- | --------------------------- | -------------------------------------------- |
| `-o`, `--output PATH`                     | `<data stem>_corrected.tsv` | below                                        |
| `--output-format {tsv,csv,parquet}`       | from the extension          | below                                        |
| `--algorithm {ComBat,limma}`              | `ComBat`                    | [Choosing an engine](Algorithms)             |
| `--combat-mode {1,2,3,4}`                 | 1                           | [ComBat](ComBat#the-four-modes)              |
| `--needed-values N`                       | 2; 1 in ComBat modes 2, 4   | [Missing values](Missing-Values)             |
| `--sort {sparsity,jaccard,seriation}`     | none                        | [Sorting and blocking](Sorting-and-Blocking) |
| `--block N`                               | none                        | [Sorting and blocking](Sorting-and-Blocking) |
| `--unique-removal`, `--no-unique-removal` | on                          | [Unique removal](Unique-Removal)             |
| `--report PATH`                           | none                        | [Correction report](Correction-Report)       |
| `--summary PATH`                          | none                        | below                                        |
| `--json`                                  | off                         | below                                        |
| `--dry-run`                               | off                         | below                                        |
| `--config PATH`                           | none                        | below                                        |
| `-v`, `--verbose`                         | off                         | below                                        |
| `-q`, `--quiet`                           | off                         | below                                        |
| `--log-file PATH`                         | `<output stem>.log`         | below                                        |
| `--no-log`                                | off                         | below                                        |
| `--version`                               |                             | prints the version                           |

## Output

Without `-o`, the corrected matrix is written next to the data file as `<data stem>_corrected.tsv`, or `.csv` or `.parquet` with `--output-format`. With `-o`, the format comes from the extension (`.csv`, `.parquet`, `.pq`, otherwise tab-separated) unless `--output-format` names one. Missing values are written as empty cells in TSV and CSV, and as nulls in Parquet. Parquet needs the `io` extra. Each output, the report and the summary included, is written to a temporary file beside it and then renamed, so a write that fails leaves no partial file and keeps any file that was there; the log file is written as the run goes.

`--report PATH` writes the per-feature report, by extension (`.csv`, `.parquet` or `.pq`, otherwise tab-separated); `--output-format` does not apply to it.

`--summary PATH` writes a JSON summary after the run: the version, the input and output paths, the settings with the `needed_values` actually used, and the feature, sample and batch counts. `--json` prints the same object to standard output and lowers the log to warnings.

No output may overwrite an input or another output: a run whose `-o`, `--summary`, `--report` or log file names the data file, the description, or another of these files stops with a usage error (exit code 2) before anything is read.

## Config files

`--config PATH` reads settings from a TOML, JSON or YAML file (`.toml`, `.json`, `.yaml`, `.yml`); YAML needs the `config` extra. Its keys are the long flag names with underscores, and each value is checked like the matching flag, so a value the flag would refuse is a usage error:

```toml
algorithm = "ComBat"
combat_mode = 1
sort = "sparsity"
block = 2
unique_removal = true
output = "corrected.tsv"
report = "report.tsv"
```

- Allowed keys: `algorithm`, `combat_mode`, `needed_values`, `sort`, `block`, `unique_removal`, `output`, `output_format`, `summary`, `report`. Any other key is a usage error.
- Flags on the command line override the file.
- Paths in the file are taken relative to the current directory, not to the file.
- A value from the file passes the same type and choice checks as its flag: `combat_mode: 7` or `output_format: xlsx` is a usage error (exit code 2) naming the key. A value the flag accepts but the library refuses, such as `block = 1`, exits with status 1, as the flag does.

## Dry run

`--dry-run` reads and validates the inputs and plans the groups exactly as a run would, sorting, blocking and unique removal included, then prints the plan and exits without correcting or writing anything. It refuses `--log-file`, `--summary`, `--json` and `--report` on the command line, even when the flag repeats the config file's value, and ignores `summary` and `report` in a config file. The input-scale warning still prints.

The plan it prints is shown on [Quick start](Quick-Start#2-check-the-plan).

## Logging

Progress goes to standard error as `INFO [harmonizepy] ...` lines. `-v` adds debug lines (affiliation counts, per-group detail); `-q` and `--json` show warnings and errors only. The input-scale warning is a Python warning, not a log line, and prints under `-q` too.

A log file with debug detail is written next to the output as `<output stem>.log`, replacing any earlier one; `--log-file PATH` moves it and `--no-log` turns it off. It is opened before the inputs are read, so a run that fails validating its input still leaves a log; a missing input file stops the run before the log is opened.

## Exit status

| Status | Meaning                                        |
| ------ | ---------------------------------------------- |
| 0      | success, or a valid dry run                    |
| 1      | input, validation, correction or output error  |
| 2      | usage error, or a missing input or config file |

- Status 1 prints one line starting `ERROR` to standard error, such as `ERROR: ...` for invalid input or `ERROR writing output to ...` when a file cannot be written.
- Status 2 covers what the argument parser refuses (an unknown choice, a value that is not a number, `-v` with `-q`), config file errors (unknown key, unsupported extension, unreadable file, a value the matching flag would refuse), dry-run conflicts, and outputs that would overwrite an input or each other. It prints the usage and a `harmonizepy: error:` line.
- A value the parser accepts but the library refuses, such as `--block 1` or `--needed-values 0`, exits with status 1.
- A summary that cannot be written prints a warning and does not change the status.

## Shell completion

With the `completion` extra installed, `harmonizepy` supports tab completion through argcomplete. Activate it as argcomplete's documentation describes for your shell, for example `eval "$(register-python-argcomplete harmonizepy)"` in bash.
