# Lab 6 Python code - Cholesterol Data

import pandas as pd
import statsmodels.formula.api as smf

# PROBLEM 2 - Cholesterol

# Load the data
data = pd.read_table("cholesterol.txt", sep="\s+")

print("=== Problem 2: Cholesterol Data ===")
print(data.head(10))
print("\nColumn names:", data.columns.tolist())

# Model A: BP ~ AGE + WGT
model_a = smf.ols('BP ~ AGE + WGT', data=data).fit()
print("\n--- Model A Summary ---")
print(model_a.summary())



# b. Predict the systolic blood pressure of a 55 year old that weighs 150 lbs
pred_bp = model_a.predict({'AGE':[55], 'WGT':[150]})
print(f"\nPredicted BP (AGE 55, WGT 150): {pred_bp.iloc[0]:.1f}")

# c. Calculate the residual for a 50-year-old man that weighs 200 pounds and has blood pressure of 123
predicted = model_a.predict({'AGE':[50], 'WGT':[200]})[0]
residual = 123 - predicted
print(f"Residual for AGE 50, WGT 200, BP 123: {residual:.1f}")


# e. Find the correlation between each pair of variables
corr_matrix = data[['AGE','WGT','CHOL','EXER','LDL']].corr()
print("\nCorrelation between variables of interest:")
print(corr_matrix)



# Model B: BP ~ AGE + WGT + CHOL
model_b = smf.ols('BP ~ AGE + WGT + CHOL', data=data).fit()
print("\n--- Model B Summary ---")
print(model_b.summary())


# Model C: BP ~ WGT + CHOL + EXER
model_c = smf.ols('BP ~ WGT + CHOL + EXER', data=data).fit()
print("\n--- Model C Summary ---")
print(model_c.summary())


# Model D: BP ~ AGE + WGT + EXER
model_d = smf.ols('BP ~ AGE + WGT + EXER', data=data).fit()
print("\n--- Model D Summary ---")
print(model_d.summary())
############
# b. Predicted BP for AGE=55, WGT=150
# Predicted BP = 128.9

# c. Residual for AGE=50, WGT=200, BP=123
# Residual = -8.0

# d. Interpretation of AGE coefficient
# Each additional year of age increases BP by 0.9541 units, holding weight constant

#e
# Evidence of multicollinearity: High correlation between CHOL and LDL (0.967)

# f. Significance of AGE and WGT in Model A
# Both significant (p < 0.05), meaning they are statistically associated with BP

# g. Model B Coefficients (BP ~ AGE + WGT + CHOL)
# Adjusted R-squared: 0.659
# Significant: AGE, WGT; CHOL not significant

# Model C Coefficients (BP ~ WGT + CHOL + EXER)
# Adjusted R-squared: 0.055
# None significant

# Model D Coefficients (BP ~ AGE + WGT + EXER)
# Adjusted R-squared: 0.644
# Significant: AGE, WGT; EXER not significant

# h. Optimal Model
# Model B is optimal: highest R-squared (0.696), AGE and WGT significant
# Limitations of other models:
# Model A: slightly lower R-squared, CHOL not included
# Model C: very low R-squared, none significant
# Model D: EXER not significant, slightly lower adjusted R-squared than B