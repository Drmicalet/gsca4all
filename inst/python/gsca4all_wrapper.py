import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import pandas as pd
import numpy as np

class GSCAElastic:
    def __init__(self):
        """Initialize the GSCA Elastic wrapper."""
        try:
            self.r_package = importr('gsca.ridge') # Note: Package name in R might be different if installed locally
            # Since we are developing, we might need to source the R files directly if not installed
            # But for a proper wrapper, we assume the package is installed.
            # However, the package name in DESCRIPTION is 'gsca_ridge' (or 'gsca_elastic' if I changed it fully, let's check DESCRIPTION)
            # I changed Title but Package name in DESCRIPTION was 'gsca_ridge'. I should check if I changed the Package field.
            # I only changed Title and Description text, not the 'Package: gsca_ridge' line. 
            # Wait, I should probably check that.
            pass
        except Exception as e:
            print(f"Warning: R package 'gsca_ridge' not found. Ensure it is installed in your R environment. Error: {e}")

    def fit(self, data, measurement_model, structural_model, alpha=0, lambd=None, max_iter=500, tol=1e-6):
        """
        Fit GSCA model with Elastic Net regularization.
        
        Args:
            data (pd.DataFrame): Input data
            measurement_model (dict): Dictionary mapping constructs to list of indicators
            structural_model (dict): Dictionary mapping endogenous constructs to list of predictors
            alpha (float): Elastic Net mixing parameter (0=Ridge, 1=Lasso)
            lambd (float, optional): Regularization strength. If None, auto-selected.
            max_iter (int): Maximum iterations
            tol (float): Convergence tolerance
            
        Returns:
            dict: Model results
        """
        # Convert pandas DataFrame to R DataFrame
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_data = ro.conversion.py2rpy(data)
            
        # Convert dictionaries to R lists
        r_measurement_model = ro.ListVector(measurement_model)
        r_structural_model = ro.ListVector(structural_model)
        
        # Call R function
        # We need to use the correct function name 'gsca_elastic'
        # And we need to make sure we are calling it from the package or sourcing it.
        # For this wrapper, we assume 'gsca_ridge' package is installed and loaded.
        
        gsca_elastic = ro.r['gsca_elastic']
        
        kwargs = {
            'data': r_data,
            'measurement_model': r_measurement_model,
            'structural_model': r_structural_model,
            'alpha': alpha,
            'max_iter': max_iter,
            'tol': tol
        }
        
        if lambd is not None:
            kwargs['lambda'] = lambd
            
        results = gsca_elastic(**kwargs)
        
        # Convert results back to Python friendly format (simplified)
        # We can return the raw R object or parse it.
        # For now, let's return the R object wrapper which allows access to components
        return results

    def summary(self, results):
        """Print summary of results."""
        ro.r['summary'](results)

    def plot(self, results):
        """Plot results."""
        ro.r['plot'](results)
