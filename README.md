# library-gsca_ridge
This package provides a complete implementation of GSCA with Ridge regularization, following the Alternating Least Squares (ALS) procedure with Ridge penalty for improved stability and reduced multicollinearity
# README.md

# GSCA with Ridge Regularization

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17155474.svg)](https://doi.org/10.5281/zenodo.17155474)

An R package implementing Generalized Structured Component Analysis (GSCA) with Ridge regularization for structural equation modeling.

## Overview

This package provides a complete implementation of GSCA with Ridge regularization, following the Alternating Least Squares (ALS) procedure with Ridge penalty for improved stability and reduced multicollinearity. The implementation was developed for analyzing data from a survey of **190 hotel staff members** who responded to questions about ICT usage in their workplaces.

## Features

- Complete GSCA implementation with Ridge regularization
- Support for both reflective and formative measurement models
- Bootstrap functionality for statistical inference
- Comprehensive reliability assessment (rho_DG, AVE)
- Discriminant validity testing
- Model fit indices (FIT, SRMR)
- S3 methods for print, summary, and plot generic functions

## Installation

```r
# Install development version from GitHub
remotes::install_github("Drmicalet/library-gsca_ridge/")

# Load the package
library(gsca_ridge)
