# HarmonizePy

HarmonizePy removes batch effects from omics matrices that have missing values, in Python, without imputing anything and without R at runtime. It runs ComBat or limma's `removeBatchEffect` inside the pipeline of HarmonizR, an R package for the same problem: each feature is corrected together with the features that have enough values in the same set of batches (after one adjustment, unique removal, that moves lone features into a shared set), and only on the values that were actually measured.

It is written for proteomics first, where a protein can be missing from a whole batch, and works on any features by samples matrix of log-scale values.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="/LangeLab/HarmonizePy/wiki/assets/batch-shift-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="/LangeLab/HarmonizePy/wiki/assets/batch-shift-light.png">
    <img alt="Four proteins measured in two batches: batch 2 reads about 1.5 units higher before correction, and the two batch means come within 0.3 of each other after harmonize()." src="/LangeLab/HarmonizePy/wiki/assets/batch-shift-light.png">
  </picture>
</p>
<p align="center"><sub><b>Figure 1.</b> Four proteins from the <a href="Quick-Start">Quick start</a> example. Each short bar is a batch mean: the dashed line between them is the batch effect, and after correction it is nearly flat while each protein keeps its own level.</sub></p>

## Why it exists

Here and elsewhere in this wiki, "I" is the author of HarmonizePy.

Labs run their pipelines in Python. Calling R from Python adds a second runtime, a second package manager, and a version pin that the Python user does not control, so I wanted the HarmonizR approach ([Voss et al. 2022](References#voss-et-al-2022)) without any of that.

HarmonizR is the reference I check HarmonizePy against, not a target it copies. Where HarmonizePy behaves differently, the difference is a decision, taken because it improves the statistics or keeps more of the user's data, and it is documented on [R compatibility](R-Compatibility).

## At a glance

- Two engines: ComBat (empirical Bayes, four modes) and limma `removeBatchEffect`, both in NumPy.
- Missing values stay missing. A feature is corrected only over the batches where it has enough values; the rest of its row comes out empty.
- Every run can report what happened to each feature: corrected, kept with its measured values and why, or emptied.
- Results are compared against committed outputs of `sva::ComBat`, `limma::removeBatchEffect` and HarmonizR 1.10.0, with the tolerance stated for each test.
- One call from Python, `harmonize("data.tsv", "batch.csv")`, or one command, `harmonizepy data.tsv batch.csv`.

## Where to start

- New to the tool: [Installation](Installation), then [Quick start](Quick-Start), which runs one small matrix from start to finish.
- New to batch correction: [Batch effects](Batch-Effects) and [Missing values](Missing-Values) explain what the tool removes and what it refuses to invent.
- Deciding settings: [Choosing an engine](Algorithms), then [ComBat](ComBat) or [limma](limma).
- Checking a run: [Correction report](Correction-Report) explains every status a feature can end with.
- Before relying on it: [Limits](Limits) lists what HarmonizePy does not do.

## References

The methods behind HarmonizePy are ComBat ([Johnson et al. 2007](References#johnson-et-al-2007)), limma ([Ritchie et al. 2015](References#ritchie-et-al-2015)) and the HarmonizR approach ([Voss et al. 2022](References#voss-et-al-2022); [Schlumbohm et al. 2025](References#schlumbohm-et-al-2025)). Every reference used in this wiki is on [References](References).
