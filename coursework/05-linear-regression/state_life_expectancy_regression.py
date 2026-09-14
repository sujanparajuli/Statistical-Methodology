import pandas as pd
import statsmodels.formula.api as smf

# PROBLEM 3 - States Data

# Load the data
data = pd.read_table("states.txt", sep="\s+")

print("\n\n=== Problem 3: States Data ===")
print(data.head(10))
print("\nColumn names:", data.columns.tolist())

# Full correlation matrix
print("\nCorrelation Matrix:")
print(round(data.corr(numeric_only=True), 2))

# Full multiple regression model (Life_exp ~ Murder + Grad + Income + Illiteracy)
model_full = smf.ols('Life_exp ~ Murder + Grad + Income + Illiteracy', data=data).fit()
print("\nFull Model Summary:")
print(model_full.summary())

# Reduced models
model_income_illiteracy = smf.ols('Life_exp ~ Income + Illiteracy', data=data).fit()
print("\nModel: Life_exp ~ Income + Illiteracy")
print(model_income_illiteracy.summary())

model_income = smf.ols('Life_exp ~ Income', data=data).fit()
print("\nModel: Life_exp ~ Income")
print(model_income.summary())

model_murder = smf.ols('Life_exp ~ Murder', data=data).fit()
print("\nModel: Life_exp ~ Murder")
print(model_murder.summary())

# Pairwise correlations (for interpretation)
print(f"\nMurder ~ Grad:      {data['Murder'].corr(data['Grad']):.2f}")
print(f"Murder ~ Income:    {data['Murder'].corr(data['Income']):.2f}")
print(f"Murder ~ Illiteracy:{data['Murder'].corr(data['Illiteracy']):.2f}")
print(f"Grad ~ Income:      {data['Grad'].corr(data['Income']):.2f}")
print(f"Grad ~ Illiteracy:  {data['Grad'].corr(data['Illiteracy']):.2f}")
print(f"Income ~ Illiteracy:{data['Income'].corr(data['Illiteracy']):.2f}")
#######
 #Optimal Model: Life_exp ~ Murder
# Reason: Single predictor Murder is highly significant (p < 0.001)
#R-squared = 0.602 higher than other. Another option for good r sq was full model but had to rule out cause income and illetracy were not significant plus multicolinearity