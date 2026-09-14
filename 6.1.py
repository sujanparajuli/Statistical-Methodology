# Lab 6 Python code

import pandas as pd
import statsmodels.formula.api as smf

# 
# PROBLEM 1 - Birthweight

# Load the data
data = pd.read_table("birthweight.txt", sep="\s+")   # or use pd.read_csv() if it's comma-separated

print("=== Problem 1: Birthweight Data ===")
print(data.head(10))
print("\nColumn names:", data.columns.tolist())

# Fit the model: BIRTH_WGT ~ GEST_AGE + AGE
model_a = smf.ols('BIRTH_WGT ~ GEST_AGE + AGE', data=data).fit()
print("\nModel Summary:")
print(model_a.summary())

# Coefficients table for Model A
coeff_a = model_a.params
se_a = model_a.bse
t_a = model_a.tvalues
p_a = model_a.pvalues
print("\nModel A Coefficients Table:")
print(pd.DataFrame({
    "Estimate": coeff_a,
    "Std. Error": se_a,
    "t value": t_a,
    "p-value": p_a
}))

print(f"\nMultiple R-squared: {model_a.rsquared:.3f}")
print(f"Adjusted R-squared: {model_a.rsquared_adj:.3f}")

# Correlation between GEST_AGE and AGE
cor_gest_age_age = data['GEST_AGE'].corr(data['AGE'])
print(f"\nCorrelation between GEST_AGE and AGE: {cor_gest_age_age:.3f}")
if abs(cor_gest_age_age) > 0.7:
    print("High correlation may indicate multicollinearity.")
else:
    print("No strong evidence of multicollinearity based on correlation.")

# Correlation matrix for Model A predictors
print("\nCorrelation matrix for Model A predictors:")
print(data[['GEST_AGE','AGE']].corr())

# Predict birth weight for 35 weeks and age 28
pred_birthweight = model_a.predict({'GEST_AGE': [35], 'AGE': [28]})
print(f"\nPredicted birthweight (35 weeks, age 28): {pred_birthweight.iloc[0]:.1f} grams")

# Residual for 3100 gram newborn at 38 weeks and mother age 30
observed = 3100
predicted = model_a.predict({'GEST_AGE': [38], 'AGE': [30]})[0]
residual = observed - predicted
print(f"\nResidual for 3100g, 38 weeks, age 30: {residual:.1f} grams")

# Interpretation of age coefficient
print(f"\nInterpretation: Each additional year in mother's age changes birthweight by {coeff_a['AGE']:.2f} grams, holding gestational age constant.")

# Check significance at 5% level
significant_vars_a = p_a[p_a < 0.05].index.tolist()
print(f"\nSignificant variables in Model A at 5%: {significant_vars_a}")

# Extend the model to include BMI
model_b = smf.ols('BIRTH_WGT ~ GEST_AGE + AGE + BMI', data=data).fit()
print("\nModel Summary:")
print(model_b.summary())

# Coefficients table for Model B
coeff_b = model_b.params
se_b = model_b.bse
t_b = model_b.tvalues
p_b = model_b.pvalues
print("\nModel B Coefficients Table:")
print(pd.DataFrame({
    "Estimate": coeff_b,
    "Std. Error": se_b,
    "t value": t_b,
    "p-value": p_b
}))

print(f"\nMultiple R-squared: {model_b.rsquared:.3f}")
print(f"Adjusted R-squared: {model_b.rsquared_adj:.3f}")

# Significant variables in Model B at 5%
significant_vars_b = p_b[p_b < 0.05].index.tolist()
print(f"\nSignificant variables in Model B at 5%: {significant_vars_b}")

# Correlation matrix for Model B predictors
print("\nCorrelation matrix for Model B predictors:")
print(data[['GEST_AGE','AGE','BMI']].corr())

# Alter the model to include DR_VISIT and not BMI
model_c = smf.ols('BIRTH_WGT ~ GEST_AGE + AGE + DR_VISIT', data=data).fit()
print("\nModel Summary:")
print(model_c.summary())

# Coefficients table for Model C
coeff_c = model_c.params
se_c = model_c.bse
t_c = model_c.tvalues
p_c = model_c.pvalues
print("\nModel C Coefficients Table:")
print(pd.DataFrame({
    "Estimate": coeff_c,
    "Std. Error": se_c,
    "t value": t_c,
    "p-value": p_c
}))

print(f"\nMultiple R-squared: {model_c.rsquared:.3f}")
print(f"Adjusted R-squared: {model_c.rsquared_adj:.3f}")

# Correlation matrix for Model C predictors
print("\nCorrelation matrix for Model C predictors:")
print(data[['GEST_AGE','AGE','DR_VISIT']].corr())

print("\nModel choice discussion:")
print("Compare R-squared, adjusted R-squared, and significance of predictors to select best model for predicting birthweight.")

###################
# a. Coefficients Table for Model A
# Intercept: -3352.39, GEST_AGE: 198.82, AGE: -32.93
# Multiple R-squared: 0.345, Adjusted R-squared: 0.336

# b. Correlation between GEST_AGE and AGE
# cor(GEST_AGE, AGE) = -0.263
# No strong evidence of multicollinearity

# c. Predicted birth weight for 35 weeks, mother age 28
# Predicted birthweight = 2684.4 grams
# This is within the range of observed gestational age and age, so not extrapolation

# d. Residual for 3100g newborn, 38 weeks, mother age 30
# Predicted birthweight = 3215.0 grams
# Residual = observed - predicted = 3100 - 3215 = -115.0 grams

# e. Interpretation of AGE coefficient
# Each additional year in mother's age decreases birthweight by 32.93 grams, holding gestational age constant

# f. Significance of AGE and GEST_AGE in Model A
# Both are significant at 5% (p-values: AGE = 0.003, GEST_AGE < 0.001)

# Model B: BIRTH_WGT ~ GEST_AGE + AGE + BMI

# g. Coefficients Table for Model B
# Intercept: -3345.92, GEST_AGE: 198.83, AGE: -32.95, BMI: -0.27
# Multiple R-squared: 0.345, Adjusted R-squared: 0.331

# h. Significant variables in Model B
# GEST_AGE and AGE are significant (p < 0.05)
# BMI is not significant (p = 0.991)

# i. Correlations between variables
# cor(GEST_AGE, AGE) = -0.263
# cor(GEST_AGE, BMI) = 0.051
# cor(AGE, BMI) = -0.153
# No strong multicollinearity detected

# j. Model choice between A and B
# Both have similar R-squared; BMI does not add significance
# Model A is simpler and adequate

# Model C: BIRTH_WGT ~ GEST_AGE + AGE + DR_VISIT

# k. Coefficients Table for Model C
# Intercept: -3330.31, GEST_AGE: 199.82, AGE: -33.18, DR_VISIT: -26.19
# Multiple R-squared: 0.346, Adjusted R-squared: 0.333
# DR_VISIT is not significant (p = 0.555)
# Model choice among A, B, C: Model A preferred for simplicity and all predictors significant