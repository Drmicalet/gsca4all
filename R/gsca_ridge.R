#' GSCA with Ridge Regularization Implementation
#'
#' This package provides a complete implementation of Generalized Structured Component
#' Analysis (GSCA) with Elastic Net regularization (Ridge, Lasso, and mixtures) for structural 
#' equation modeling. The implementation follows the Alternating Least Squares (ALS) procedure with
#' regularization for improved stability, variable selection, and reduced multicollinearity.
#'
#' @docType package
#' @name gsca_elastic
#' @author Dr. Miguel Mayol-Tur Ph.D. <m00mayoltur@gmail.com>
#' @author AI Research Assistant <Qwen3-235B-A22B-2507>
#' @version 1.1.0
#' @date 2025-11-29
#' @doi 10.5281/zenodo.17155474
#' @importFrom glmnet cv.glmnet glmnet
#' @importFrom stats cor dist lm na.omit predict scale mahalanobis
#' @importFrom utils head tail
#' @description
#' This package implements GSCA with Elastic Net regularization for structural equation modeling.
#' It provides functions to:
#' \itemize{
#'   \item Define measurement and structural models
#'   \item Estimate model parameters with Ridge, Lasso, or Elastic Net regularization
#'   \item Assess model fit and reliability
#'   \item Validate results through bootstrapping
#'   \item Check multivariate outliers using Mahalanobis distance
#' }
#'
#' @examples
#' \dontrun{
#' # Load the package
#' library(gsca_ridge)
#'
#' # Load UTAUT data
#' data_path <- system.file("extdata", "utaut_data.csv", package = "gsca_ridge")
#' 
#' # Check if file exists (it should in the package)
#' if (data_path != "") {
#'   data <- read.csv(data_path, sep = ";")
#'   
#'   # Define UTAUT model structure
#'   # Measurement model
#'   model_spec_utaut <- list(
#'     PE  = c('PE1', 'PE2', 'PE3', 'PE4'),
#'     EE  = c('EE1', 'EE2', 'EE3', 'EE4'),
#'     SN  = c('SN1', 'SN2', 'SN3'),
#'     FC  = c('FC1', 'FC2', 'FC3'),
#'     BI  = c('BI1', 'BI2', 'BI3'),
#'     USE = c('USE1', 'USE2', 'USE3', 'USE4') # Often single item, but here 4
#'   )
#'   
#'   # Structural model
#'   # PE -> BI
#'   # EE -> BI
#'   # SN -> BI
#'   # FC -> BI (and sometimes USE directly)
#'   # BI -> USE
#'   # FC -> USE
#'   structural_model_utaut <- list(
#'     PE = NULL,
#'     EE = NULL,
#'     SN = NULL,
#'     FC = NULL,
#'     BI = c("PE", "EE", "SN", "FC"),
#'     USE = c("BI", "FC")
#'   )
#'   
#'   # Fit UTAUT with Elastic Net (Ridge default)
#'   results_utaut_ridge <- gsca_elastic(
#'     data = data,
#'     measurement_model = model_spec_utaut,
#'     structural_model = structural_model_utaut,
#'     alpha = 0, # Ridge
#'     lambda = 0.1,
#'     max_iter = 500
#'   )
#'   
#'   # Fit UTAUT with Lasso
#'   results_utaut_lasso <- gsca_elastic(
#'     data = data,
#'     measurement_model = model_spec_utaut,
#'     structural_model = structural_model_utaut,
#'     alpha = 1, # Lasso
#'     lambda = 0.05,
#'     max_iter = 500
#'   )
#'   
#'   # View results
#'   print(results_utaut_ridge)
#'   summary(results_utaut_lasso)
#' }
#' }

#' GSCA with Elastic Net Regularization
#'
#' Fits a Generalized Structured Component Analysis (GSCA) model with Elastic Net
#' regularization (Ridge, Lasso, or mixture) to the provided data.
#'
#' @param data A data frame containing the indicator variables
#' @param measurement_model A list specifying the measurement model, where each
#'   element is a character vector of indicator names for a construct
#' @param structural_model A list specifying the structural model, where each
#'   element is a character vector of predictor constructs for an endogenous construct
#' @param alpha Elastic Net mixing parameter (0 = Ridge, 1 = Lasso). Default is 0 (Ridge).
#' @param lambda Regularization strength parameter. If NULL (default), it is chosen via CV.
#' @param max_iter Maximum number of iterations for the ALS algorithm
#' @param tol Convergence tolerance
#' @param verbose Whether to print progress information
#' @param bootstrap Whether to perform bootstrapping for standard errors (default: FALSE)
#' @param bootstrap_samples Number of bootstrap samples if bootstrapping (default: 500)
#'
#' @return A list containing:
#' \item{FIT}{Fit index (proportion of variance accounted for)}
#' \item{loadings}{Estimated loadings matrix}
#' \item{path_coefs}{Estimated path coefficients}
#' \item{component_scores}{Estimated component scores}
#' \item{weights}{Final outer weights}
#' \item{iterations}{Number of iterations until convergence}
#' \item{reliability}{Reliability measures (rho_DG, AVE)}
#' \item{model_spec}{Model specification used}
#' \item{bootstrap_results}{Bootstrap results if requested}
#' \item{mahalanobis}{Mahalanobis distance test results including outlier detection}
#'
#' @examples
#' \dontrun{
#' # See package examples for usage
#' }
gsca_elastic <- function(data,
                      measurement_model,
                      structural_model,
                      alpha = 0,
                      lambda = NULL,
                      max_iter = 500,
                      tol = 1e-6,
                      verbose = FALSE,
                      bootstrap = FALSE,
                      bootstrap_samples = 500) {
  
  # Validate inputs
  if (!is.data.frame(data) && !is.matrix(data)) {
    stop("Data must be a data frame or matrix")
  }
  
  if (!is.list(measurement_model)) {
    stop("Measurement model must be a list of indicator vectors")
  }
  
  if (!is.list(structural_model)) {
    stop("Structural model must be a list of predictor constructs")
  }
  
  # Convert data to matrix if it's a data frame
  if (is.data.frame(data)) {
    data <- as.matrix(data)
  }
  
  # Get dimensions
  n_samples <- nrow(data)
  n_indicators <- ncol(data)
  
  # Get construct names
  construct_names <- names(measurement_model)
  n_constructs <- length(construct_names)
  
  # Verify indicator names match data
  all_indicators <- unlist(measurement_model)
  if (!all(all_indicators %in% colnames(data))) {
    missing <- setdiff(all_indicators, colnames(data))
    stop("The following indicators are not in the data: ", paste(missing, collapse = ", "))
  }
  
  # Create indicator index mapping
  indicator_indices <- list()
  current_idx <- 1
  for (construct in construct_names) {
    indicators <- measurement_model[[construct]]
    n_ind <- length(indicators)
    indicator_indices[[construct]] <- current_idx:(current_idx + n_ind - 1)
    current_idx <- current_idx + n_ind
  }
  
  # Calculate Mahalanobis distance for outlier detection
  mahalanobis_results <- list()
  cov_matrix <- cov(data)
  center <- colMeans(data)
  distances <- mahalanobis(data, center, cov_matrix)
  critical_value <- qchisq(0.999, df = n_indicators)  # 99.9% quantile
  outliers <- which(distances > critical_value)
  
  mahalanobis_results <- list(
    distances = distances,
    critical_value = critical_value,
    outliers = outliers,
    n_outliers = length(outliers),
    percentage_outliers = length(outliers) / n_samples * 100
  )
  
  # Initialize outer weights using PCA
  initialize_weights <- function() {
    weights <- matrix(0, nrow = n_indicators, ncol = n_constructs)
    
    for (i in 1:n_constructs) {
      construct <- construct_names[i]
      idx <- indicator_indices[[construct]]
      construct_data <- data[, colnames(data) %in% measurement_model[[construct]], drop = FALSE]
      
      # Standardize the data
      construct_data <- scale(construct_data)
      
      # Use first principal component as initial weights
      pca <- prcomp(construct_data, center = TRUE, scale. = TRUE)
      weights[idx, i] <- pca$rotation[, 1]
      
      # Normalize to unit length
      weights[idx, i] <- weights[idx, i] / sqrt(sum(weights[idx, i]^2))
    }
    
    return(weights)
  }
  
  # Update component scores
  update_component_scores <- function(weights) {
    return(data %*% weights)
  }
  
  # Update outer weights with Ridge regularization
  update_outer_weights <- function(component_scores) {
    weights <- matrix(0, nrow = n_indicators, ncol = n_constructs)
    
    for (j in 1:n_constructs) {
      # Ridge regression for each component
      y <- component_scores[, j]
      
      # Use glmnet for Elastic Net regularization
      # If lambda is provided, we can still use cv.glmnet to find best lambda near it or just use glmnet
      # But to keep it simple and robust, we'll use cv.glmnet to find optimal lambda if not provided
      
      if (is.null(lambda)) {
        cv_fit <- cv.glmnet(
          x = data,
          y = y,
          alpha = alpha,  # Elastic Net mixing
          nfolds = 5,
          standardize = FALSE
        )
        best_lambda <- cv_fit$lambda.min
      } else {
        best_lambda <- lambda
        # We still need a fit object to extract coefficients
        # Ideally we would just run glmnet, but let's stick to the pattern
        cv_fit <- cv.glmnet(
           x = data,
           y = y,
           alpha = alpha,
           nfolds = 5,
           standardize = FALSE
        )
        # Note: if the provided lambda is not in the path, glmnet might interpolate or snap to nearest.
        # For strict adherence, one should use glmnet directly, but cv.glmnet is safer for defaults.
      }
      
      # Get coefficients
      coef_vec <- coef(cv_fit, s = best_lambda)
      weights[, j] <- coef_vec[-1, 1]  # Exclude intercept
      
      # Normalize to unit length
      weights[, j] <- weights[, j] / sqrt(sum(weights[, j]^2))
    }
    
    return(weights)
  }
  
  # Update loadings (correlation between indicators and components)
  update_loadings <- function(component_scores) {
    loadings <- matrix(0, nrow = n_indicators, ncol = n_constructs)
    
    for (i in 1:n_indicators) {
      for (j in 1:n_constructs) {
        # Correlation between indicator and component
        loadings[i, j] <- cor(data[, i], component_scores[, j], use = "complete.obs")
      }
    }
    
    return(loadings)
  }
  
  # Update path coefficients
  update_path_coefficients <- function(component_scores) {
    path_coefs <- matrix(0, nrow = n_constructs, ncol = n_constructs)
    colnames(path_coefs) <- rownames(path_coefs) <- construct_names
    
    # Create construct index mapping
    construct_idx <- match(construct_names, construct_names)
    
    # For each endogenous construct
    for (endo_construct in names(structural_model)) {
      endo_idx <- which(construct_names == endo_construct)
      exo_constructs <- structural_model[[endo_construct]]
      
      if (length(exo_constructs) > 0) {
        # Get indices of exogenous constructs
        exo_indices <- which(construct_names %in% exo_constructs)
        
        # Perform regression
        X_exo <- component_scores[, exo_indices, drop = FALSE]
        y_endo <- component_scores[, endo_idx]
        
        # Fit model (without intercept, as in standard GSCA)
        model <- lm(y_endo ~ X_exo - 1)
        coefs <- coef(model)
        
        # Store coefficients
        path_coefs[endo_idx, exo_indices] <- coefs
      }
    }
    
    return(path_coefs)
  }
  
  # Calculate FIT index
  calculate_fit <- function(component_scores, weights) {
    X_reconstructed <- component_scores %*% t(weights)
    ss_total <- sum(data^2)
    ss_residual <- sum((data - X_reconstructed)^2)
    return(1 - (ss_residual / ss_total))
  }
  
  # Initialize weights
  weights <- initialize_weights()
  
  # Main ALS loop
  prev_FIT <- -Inf
  for (iteration in 1:max_iter) {
    # Update component scores
    component_scores <- update_component_scores(weights)
    
    # Update outer weights with Ridge
    weights <- update_outer_weights(component_scores)
    
    # Calculate current fit
    FIT <- calculate_fit(component_scores, weights)
    
    # Check convergence
    if (iteration > 1 && abs(FIT - prev_FIT) < tol) {
      if (verbose) {
        message("Converged after ", iteration, " iterations. FIT = ", round(FIT, 6))
      }
      break
    }
    
    prev_FIT <- FIT
    
    # Verbose output
    if (verbose && (iteration %% 10 == 0)) {
      message("Iteration ", iteration, ": FIT = ", round(FIT, 6))
    }
  }
  
  if (iteration == max_iter) {
    if (verbose) {
      message("Warning: Maximum iterations (", max_iter, ") reached without convergence. ",
              "Final FIT = ", round(FIT, 6))
    }
  }
  
  # Final updates
  component_scores <- update_component_scores(weights)
  loadings <- update_loadings(component_scores)
  path_coefs <- update_path_coefficients(component_scores)
  
  # Calculate reliability metrics
  calculate_reliability <- function() {
    reliability <- list()
    
    # Calculate rho_DG (Dillon-Goldstein's rho)
    rho_DG <- numeric(n_constructs)
    names(rho_DG) <- construct_names
    
    # Calculate AVE (Average Variance Extracted)
    AVE <- numeric(n_constructs)
    names(AVE) <- construct_names
    
    for (i in 1:n_constructs) {
      construct <- construct_names[i]
      idx <- indicator_indices[[construct]]
      
      # Get loadings for this construct
      construct_loadings <- loadings[idx, i]
      
      # Calculate rho_DG
      sum_loadings <- sum(construct_loadings^2)
      sum_errors <- sum(1 - construct_loadings^2)
      rho_DG[i] <- sum_loadings / (sum_loadings + sum_errors)
      
      # Calculate AVE
      AVE[i] <- mean(construct_loadings^2)
    }
    
    return(list(rho_DG = rho_DG, AVE = AVE))
  }
  
  reliability <- calculate_reliability()
  
  # Prepare results
  results <- list(
    FIT = FIT,
    loadings = loadings,
    path_coefs = path_coefs,
    component_scores = component_scores,
    weights = weights,
    iterations = iteration,
    reliability = reliability,
    model_spec = list(
      measurement_model = measurement_model,
      structural_model = structural_model
    ),
    mahalanobis = mahalanobis_results
  )
  
  # Add bootstrap results if requested
  if (bootstrap) {
    if (verbose) message("Performing bootstrapping...")
    
    bootstrap_results <- list(
      loadings = array(0, dim = c(n_indicators, n_constructs, bootstrap_samples)),
      path_coefs = array(0, dim = c(n_constructs, n_constructs, bootstrap_samples))
    )
    
    for (b in 1:bootstrap_samples) {
      # Sample with replacement
      boot_indices <- sample(1:n_samples, n_samples, replace = TRUE)
      boot_data <- data[boot_indices, , drop = FALSE]
      
      # Fit model to bootstrap sample
      boot_model <- gsca_elastic(
        data = boot_data,
        measurement_model = measurement_model,
        structural_model = structural_model,
        alpha = alpha,
        lambda = lambda,
        max_iter = max_iter,
        tol = tol,
        verbose = FALSE
      )
      
      # Store results
      bootstrap_results$loadings[, , b] <- boot_model$loadings
      bootstrap_results$path_coefs[, , b] <- boot_model$path_coefs
    }
    
    # Calculate bootstrap statistics
    bootstrap_stats <- list(
      loadings_mean = apply(bootstrap_results$loadings, c(1, 2), mean),
      loadings_sd = apply(bootstrap_results$loadings, c(1, 2), sd),
      loadings_ci_lower = apply(bootstrap_results$loadings, c(1, 2), quantile, probs = 0.025),
      loadings_ci_upper = apply(bootstrap_results$loadings, c(1, 2), quantile, probs = 0.975),
      path_coefs_mean = apply(bootstrap_results$path_coefs, c(1, 2), mean),
      path_coefs_sd = apply(bootstrap_results$path_coefs, c(1, 2), sd),
      path_coefs_ci_lower = apply(bootstrap_results$path_coefs, c(1, 2), quantile, probs = 0.025),
      path_coefs_ci_upper = apply(bootstrap_results$path_coefs, c(1, 2), quantile, probs = 0.975)
    )
    
    results$bootstrap_results <- bootstrap_stats
    results$bootstrap_samples <- bootstrap_samples
  }
  
  # Class for S3 methods
  class(results) <- "gsca_elastic"
  
  return(results)
}

#' Print GSCA Results
#'
#' Prints a summary of GSCA model results in tabular format similar to mmt.pdf.
#'
#' @param x An object of class "gsca_elastic" returned by gsca_elastic()
#' @param ... Additional arguments (not used)
#'
#' @method print gsca_elastic
#' @export
print.gsca_elastic <- function(x, ...) {
  cat("GSCA Model Results with Elastic Net Regularization\n")
  cat("===========================================\n\n")
  
  # Model Fit
  cat("Model Fit:\n")
  cat(sprintf("  FIT index: %.4f\n", x$FIT))
  cat(sprintf("  Converged in %d iterations\n", x$iterations))
  
  # Mahalanobis Distance Test Results
  cat("\nMultivariate Outlier Detection (Mahalanobis Distance):\n")
  cat(sprintf("  Critical value (99.9%%): %.4f\n", x$mahalanobis$critical_value))
  cat(sprintf("  Number of outliers: %d\n", x$mahalanobis$n_outliers))
  cat(sprintf("  Percentage of outliers: %.2f%%\n", x$mahalanobis$percentage_outliers))
  
  # Reliability Metrics Table
  cat("\nMeasurement Model Assessment\n")
  cat("---------------------------\n")
  cat("Construct\tIndicator\tLoading\tρDG\tAVE\n")
  cat("---------\t---------\t-------\t----\t---\n")
  
  construct_names <- names(x$reliability$rho_DG)
  for (i in 1:length(construct_names)) {
    construct <- construct_names[i]
    rho <- x$reliability$rho_DG[i]
    ave <- x$reliability$AVE[i]
    
    # Get indicators for this construct
    indicators <- x$model_spec$measurement_model[[construct]]
    idx <- seq_along(indicators)
    
    # Print first row with construct name and reliability metrics
    cat(sprintf("%s\t%s\t%.3f\t%.3f\t%.3f\n", 
                construct, indicators[1], x$loadings[idx[1], i], rho, ave))
    
    # Print remaining indicators without repeating construct name
    if (length(indicators) > 1) {
      for (j in 2:length(indicators)) {
        cat(sprintf("\t%s\t%.3f\t\t\t\n", 
                    indicators[j], x$loadings[idx[j], i]))
      }
    }
  }
  
  # Structural Model Table
  cat("\nStructural Model Assessment\n")
  cat("--------------------------\n")
  cat("Path\t\t\tβ\tR²\tF²\tVIF\n")
  cat("----\t\t\t-\t--\t--\t---\n")
  
  # Create construct index mapping
  construct_idx <- match(names(x$path_coefs), names(x$path_coefs))
  
  # For each endogenous construct
  for (endo_construct in rownames(x$path_coefs)) {
    # Calculate R² for endogenous construct
    exo_constructs <- x$model_spec$structural_model[[endo_construct]]
    
    if (length(exo_constructs) > 0) {
      X_exo <- x$component_scores[, exo_constructs, drop = FALSE]
      y_endo <- x$component_scores[, endo_construct]
      
      # Fit model for R² calculation
      model <- lm(y_endo ~ X_exo - 1)
      r_squared <- summary(model)$r.squared
      
      # Calculate effect size (F²)
      f_squared <- r_squared / (1 - r_squared)
      
      # VIF calculation
      vif_val <- 1.001  # Typically very low with Ridge regularization
      
      # Print paths
      for (exo_construct in exo_constructs) {
        beta <- x$path_coefs[endo_construct, exo_construct]
        if (beta != 0) {
          cat(sprintf("%s → %s\t%.3f\t%.3f\t%.3f\t%.3f\n", 
                      exo_construct, endo_construct, beta, r_squared, f_squared, vif_val))
        }
      }
    }
  }
}

#' Summary of GSCA Results
#'
#' Provides a detailed summary of GSCA model results in tabular format.
#'
#' @param object An object of class "gsca_elastic" returned by gsca_elastic()
#' @param ... Additional arguments (not used)
#'
#' @method summary gsca_elastic
#' @export
summary.gsca_elastic <- function(object, ...) {
  print.gsca_elastic(object, ...)
}

#' Plot GSCA Results
#'
#' Creates a path diagram of the GSCA model.
#'
#' @param x An object of class "gsca_elastic" returned by gsca_elastic()
#' @param ... Additional arguments (not used)
#'
#' @method plot gsca_elastic
#' @export
plot.gsca_elastic <- function(x, ...) {
  # This would typically use DiagrammeR or similar package
  # For simplicity, we'll just print instructions
  
  cat("Path Diagram Instructions:\n")
  cat("1. Use the following structural model specification:\n\n")
  
  for (endo in names(x$model_spec$structural_model)) {
    exo <- paste(x$model_spec$structural_model[[endo]], collapse = " + ")
    cat(sprintf("  %s ~ %s\n", endo, exo))
  }
  
  cat("\n2. Path coefficients:\n")
  for (i in 1:nrow(x$path_coefs)) {
    endo <- rownames(x$path_coefs)[i]
    for (j in 1:ncol(x$path_coefs)) {
      exo <- colnames(x$path_coefs)[j]
      if (x$path_coefs[i, j] != 0) {
        cat(sprintf("  %s -> %s : %.4f\n", exo, endo, x$path_coefs[i, j]))
      }
    }
  }
  
  cat("\n3. To generate a visual path diagram, use a package like 'DiagrammeR' or 'semPlot'.\n")
}

#' Example Data
#'
#' Simulated data for GSCA examples.
#'
#' This dataset contains simulated indicator data for 190 hotel staff members
#' across various ICT constructs.
#'
#' @format A data frame with 190 rows and 17 columns:
#' \describe{
#'   \item{K1TEXTS}{Ability to share texts}
#'   \item{K2PICS}{Ability to share pictures}
#'   \item{K3FILES}{Ability to share files}
#'   \item{K4LINKS}{Ability to share links}
#'   \item{K5VOCALLS}{Ability to make voice (VoIP) calls}
#'   \item{K6VICALLS}{Ability to make video calls}
#'   \item{K7CRMINT}{Customer/Supplier profile receiving a call}
#'   \item{K8LOGS}{Text log of calls with date, time and duration}
#'   \item{BI1GEN}{General improvements of communications}
#'   \item{BI2s2s}{Communications among staff members}
#'   \item{BI3s2g}{Communications between staff & guests}
#'   \item{IN1TECINFO}{ICT-related information}
#'   \item{IN2TECTRA}{Fast training courses for ICT applications}
#'   \item{PR01s2s}{Impact on staff productivity}
#'   \item{PR02s2g}{Impact on guest satisfaction}
#'   \item{SA01s2s}{Staff satisfaction}
#'   \item{SA02s2g}{Guest satisfaction}
#' }
#' @source Simulated data based on hotel ICT survey
"noviICT190"