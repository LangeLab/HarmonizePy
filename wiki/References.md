# References

The papers and packages this wiki relies on, grouped by what they are used for. Pages cite them as author and year, linked to the entry here. Each entry gives the full reference and its DOI.

## Methods

### Johnson et al. 2007

Johnson WE, Li C, Rabinovic A. Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics* 8(1):118-127, 2007. doi:[10.1093/biostatistics/kxj037](https://doi.org/10.1093/biostatistics/kxj037)

The ComBat model and its parametric and non-parametric empirical Bayes estimates. Cited on [ComBat](ComBat) and [Choosing an engine](Algorithms).

### Leek et al. 2012

Leek JT, Johnson WE, Parker HS, Jaffe AE, Storey JD. The sva package for removing batch effects and other unwanted variation in high-throughput experiments. *Bioinformatics* 28(6):882-883, 2012. doi:[10.1093/bioinformatics/bts034](https://doi.org/10.1093/bioinformatics/bts034)

The `sva` package, whose `ComBat` function HarmonizePy is checked against, including its `par.prior` and `mean.only` switches. Cited on [ComBat](ComBat) and [R compatibility](R-Compatibility).

### Ritchie et al. 2015

Ritchie ME, Phipson B, Wu D, Hu Y, Law CW, Shi W, Smyth GK. limma powers differential expression analyses for RNA-sequencing and microarray studies. *Nucleic Acids Research* 43(7):e47, 2015. doi:[10.1093/nar/gkv007](https://doi.org/10.1093/nar/gkv007)

The reference for the `limma` package, which provides `removeBatchEffect`. The function's own documentation cites no separate paper. Cited on [limma](limma).

## HarmonizR

### Voss et al. 2022

Voss H, Schlumbohm S, Barwikowski P, Wurlitzer M, Dottermusch M, Neumann P, Schlueter H, Neumann JE, Krisp C. HarmonizR enables data harmonization across independent proteomic datasets with appropriate handling of missing values. *Nature Communications* 13:3523, 2022. doi:[10.1038/s41467-022-31007-x](https://doi.org/10.1038/s41467-022-31007-x)

The HarmonizR approach: splitting a matrix by missing-value pattern and correcting each part with ComBat or limma, without imputation. Cited on [Home](Home), [Missing values](Missing-Values) and [Pipeline](Pipeline).

### Schlumbohm et al. 2025

Schlumbohm S, Neumann JE, Neumann P. HarmonizR: blocking and singular feature data adjustment improve runtime efficiency and data preservation. *BMC Bioinformatics* 26:47, 2025. doi:[10.1186/s12859-025-06073-9](https://doi.org/10.1186/s12859-025-06073-9)

Blocking, batch sorting and unique removal in HarmonizR. Cited on [Unique removal](Unique-Removal) and [Sorting and blocking](Sorting-and-Blocking).

### HarmonizR 1.10.0

Schlumbohm S, Neumann JE, Neumann P. HarmonizR, Bioconductor package, version 1.10.0. [Bioconductor](https://bioconductor.org/packages/release/bioc/html/HarmonizR.html); source at [HSU-HPC/HarmonizR](https://github.com/HSU-HPC/HarmonizR).

The release HarmonizePy's pipeline tests compare against. Cited on [R compatibility](R-Compatibility).

## Batch effects and missing values

### Leek et al. 2010

Leek JT, Scharpf RB, Bravo HC, Simcha D, Langmead B, Johnson WE, Geman D, Baggerly K, Irizarry RA. Tackling the widespread and critical impact of batch effects in high-throughput data. *Nature Reviews Genetics* 11(10):733-739, 2010. doi:[10.1038/nrg2825](https://doi.org/10.1038/nrg2825)

How common batch effects are, and how they can outweigh the biology a study measures. Cited on [Batch effects](Batch-Effects).

### Cuklina et al. 2021

Cuklina J, Lee CH, Williams EG, Sajic T, Collins BC, Rodriguez Martinez M, Sharma VS, Wendt F, Goetze S, Keele GR, Wollscheid B, Aebersold R, Pedrioli PGA. Diagnostics and correction of batch effects in large-scale proteomic studies: a tutorial. *Molecular Systems Biology* 17(8):e10240, 2021. doi:[10.15252/msb.202110240](https://doi.org/10.15252/msb.202110240)

Sources of batch effects in proteomics, and how to diagnose them. Cited on [Batch effects](Batch-Effects).

### Rocke and Durbin 2001

Rocke DM, Durbin B. A model for measurement error for gene expression arrays. *Journal of Computational Biology* 8(6):557-569, 2001. doi:[10.1089/106652701753307485](https://doi.org/10.1089/106652701753307485)

Measurement error that grows with the signal, the reason for working on a log scale. Cited on [Batch effects](Batch-Effects).

### Nygaard et al. 2016

Nygaard V, Rodland EA, Hovig E. Methods that remove batch effects while retaining group differences may lead to exaggerated confidence in downstream analyses. *Biostatistics* 17(1):29-39, 2016. doi:[10.1093/biostatistics/kxv027](https://doi.org/10.1093/biostatistics/kxv027)

What batch correction does when batches and conditions are unevenly mixed. Cited on [Batch effects](Batch-Effects).

### Lazar et al. 2016

Lazar C, Gatto L, Ferro M, Bruley C, Burger T. Accounting for the multiple natures of missing values in label-free quantitative proteomics data sets to compare imputation strategies. *Journal of Proteome Research* 15(4):1116-1125, 2016. doi:[10.1021/acs.jproteome.5b00981](https://doi.org/10.1021/acs.jproteome.5b00981)

Why values are missing in label-free proteomics: below detection (missing not at random) or by chance (missing completely at random). Cited on [Missing values](Missing-Values).

### Du et al. 2010

Du P, Zhang X, Huang CC, Jafari N, Kibbe WA, Hou L, Lin SM. Comparison of Beta-value and M-value methods for quantifying methylation levels by microarray analysis. *BMC Bioinformatics* 11:587, 2010. doi:[10.1186/1471-2105-11-587](https://doi.org/10.1186/1471-2105-11-587)

M-values, the log-ratio scale for methylation proportions. Cited on [Input format](Input-Format).
