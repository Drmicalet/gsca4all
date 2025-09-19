# GSCA with Ridge Regularization

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17155474.svg)](https://doi.org/10.5281/zenodo.17155474)

An R package implementing Generalized Structured Component Analysis (GSCA) with Ridge regularization for structural equation modeling in hospitality research.

## Overview

This package provides a complete implementation of GSCA with Ridge regularization, following the Alternating Least Squares (ALS) procedure with Ridge penalty for improved stability and reduced multicollinearity. The implementation was developed for analyzing data from a survey of **190 hotel staff members** who responded to questions about ICT usage in their workplaces.

## Features

- Complete GSCA implementation with Ridge regularization
- Support for both reflective and formative measurement models  
- Bootstrap functionality for statistical inference
- Comprehensive reliability assessment (rho_DG, AVE)
- Discriminant validity testing
- Model fit indices (FIT, SRMR)
- Multivariate outlier detection using Mahalanobis distance
- S3 methods for print, summary, and plot generic functions
- Tabular output format matching academic standards as used in the thesis revisited:
- **Do ICT's deep use increase hotel's staff productivity and guest satisfaction?**
- https://www.researchgate.net/publication/371789905_Do_ICT's_deep_use_increase_hotel's_staff_productivity_and_guest_satisfaction 
- DOI: [10.31428/10317/12327](http://dx.doi.org/10.31428/10317/12327)

## Installation

```r
# Install development version from GitHub
remotes::install_github("Drmicalet/library-gsca_ridge")

# Load the package
library(gsca_ridge)
```

## Usage Example

```R
# Load your data
data <- read.csv("noviICT190.csv", row.names = 1)

# Define model structure for Model A+
model_spec_Aplus <- list(
  Knowledge = c('K1TEXTS', 'K2PICS', 'K3FILES', 'K4LINKS', 
               'K5VOCALLS', 'K6VICALLS', 'K7CRMINT', 'K8LOGS'),
  Improvement = c('BI1GEN', 'BI2s2s', 'BI3s2g'),
  Information = c('IN1TECINFO', 'IN2TECTRA'),
  Productivity = c('PR01s2s', 'PR02s2g'),
  Satisfaction = c('SA01s2s', 'SA02s2g')
)

structural_model_Aplus <- list(
  Improvement = "Knowledge",
  Productivity = c("Improvement", "Information"),
  Satisfaction = c("Improvement", "Information")
)

# Fit Model A+ with Ridge regularization
results_Aplus <- gsca_ridge(
  data = data,
  measurement_model = model_spec_Aplus,
  structural_model = structural_model_Aplus,
  alpha = 0.1,
  max_iter = 500,
  tol = 1e-6
)

# View results
print(results_Aplus)
summary(results_Aplus)
```



## Output Format

The package produces publication-ready tables in the style of the thesis that originated it, including:



### Measurement Model Assessment

| CONSTRUCT | INDICATOR | LOADING | ΡDG   | AVE   |
| --------- | --------- | ------- | ----- | ----- |
| Knowledge | K1TEXTS   | 0.821   | 0.821 | 0.514 |
|           | K2PICS    | 0.853   |       |       |
|           | K3FILES   | 0.789   |       |       |

### Structural Model Assessment

| PATH                    | Β     | R²    | F²    | VIF   |
| ----------------------- | ----- | ----- | ----- | ----- |
| Knowledge → Improvement | 0.666 | 0.444 | 0.800 | 1.001 |

### Validity and Reliability Criteria

| METRIC   | ACCEPTABLE | GOOD   | SOURCE                    |
| -------- | ---------- | ------ | ------------------------- |
| Loadings | > 0.7      | > 0.8  | Fornell & Larcker (1981)  |
| ρDG      | > 0.7      | > 0.8  | Dillon & Goldstein (1987) |
| AVE      | > 0.5      | > 0.6  | Fornell & Larcker (1981)  |
| FIT      | > 0.85     | > 0.90 | Tenenhaus et al. (2005)   |
| SRMR     | < 0.08     | < 0.05 | Hu & Bentler (1999)       |

## Mahalanobis Distance Outlier Detection

The package includes multivariate outlier detection using Mahalanobis distance:

-    Critical value at 99.9% confidence level
-    Identification of outlier cases
-    Percentage of outliers in sample
-    Automatic removal option for sensitivity analysis

## Citation

To cite this package in publications:

```markdown
Mayol-Tur, M., (2025). gsca_ridge: Generalized Structured Component Analysis with Ridge Regularization (Version 1.0.0) [Computer software]. https://doi.org/10.5281/zenodo.17155474 https://github.com/Drmicalet/library-gsca_ridge/
```

Mayol-Tur, M., (2025). gsca_ridge: Generalized Structured Component Analysis with Ridge Regularization (Version 1.0.0) [Computer software]. 

https://doi.org/10.5281/zenodo.17155474 https://github.com/Drmicalet/library-gsca_ridge/

## Author

**Dr. Miguel Mayol-Tur Ph.D.**
Email: [m00mayoltur@gmail.com](mailto:m00mayoltur@gmail.com)
Affiliation: Technical University of Cartagena (UPCT), Spain

This package was developed as part of a paper revisiting with IA the doctoral thesis "Do ICT's deep use increase hotel's staff productivity and guest satisfaction?" supervised by Dr. Andrés Artal Tur at the Technical University of Cartagena, in order to compare human vs IA, and the IA needed to create this new package to try to improve previous human model. 



Assisted mainly by Qwen3-235B-A22B-2507 after it invented a non existing library to code the "better model than the human did", and faked its results, after confronting how did it made the calculations if the library does not exist yet, that now is this one.

## License

GNU AFFERO GENERAL PUBLIC LICENSE Version 3, 19 November 2007

Copyright (C) 2007 Free Software Foundation, Inc. https://fsf.org/ Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed. ```

