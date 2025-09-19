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