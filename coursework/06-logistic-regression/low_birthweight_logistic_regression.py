import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# load data
data = pd.read_csv("lowbirthweight.txt", sep="\s+")

# preview data
print("data preview:")
print(data.head())

# column names
print("\ncolumn names:")
print(data.columns.tolist())

# basic statistics
print("\nbasic statistics:")
print(data.describe())

# model 1: LOW ~ SMOKE
model1 = smf.logit('LOW ~ SMOKE', data=data).fit()

print(model1.summary())
print("AIC =", model1.aic)

params1 = model1.params
print("prediction equation:")


# model 2: LOW ~ AGE
model2 = smf.logit('LOW ~ AGE', data=data).fit()

print(model2.summary())
print("AIC =", model2.aic)

params2 = model2.params
print("prediction equation:")

# plot AGE vs LOW with logistic curve
plt.scatter(data['AGE'], data['LOW'],
            alpha=0.7,
            edgecolor='navy',
            color='lightblue',
            s=60)

age_range = np.linspace(data['AGE'].min(), data['AGE'].max(), 200)
pred_df2 = pd.DataFrame({'AGE': age_range})
pred_probs2 = model2.predict(pred_df2)

plt.plot(age_range, pred_probs2,
         color='red',
         linewidth=2.8)

plt.xlabel('AGE')
plt.ylabel('LOW')
plt.title('logistic regression: LOW ~ AGE')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# model 3: LOW ~ HT
model3 = smf.logit('LOW ~ HT', data=data).fit()

print(model3.summary())
print("AIC =", model3.aic)

params3 = model3.params
print("prediction equation:")

#(a) SMOKE logistic regression
# Model: LOW ~ SMOKE
# log(p/(1-p)) = -1.0871 + 0.7041 * SMOKE
# Smoking increases the probability of low birth weight.
# Smokers have higher risk than non-smokers.

#(b) 95% CI and odds ratio for SMOKE
# Odds ratio = exp(0.7041)
# 95% CI = [1.08, 3.78]
# CI does NOT include 1 → SMOKE is statistically significant
# Smoking has a significant effect on low birth weight.

#(c) Effect of smoking
# b1 = 0.7041 (positive)
# Smoking increases the risk of low birth weight.
# Odds ratio > 1 confirms increased risk.

#(d) AGE logistic regression
# Model: LOW ~ AGE
# log(p/(1-p)) = 0.3846 - 0.0512 * AGE
# AGE has a negative effect (older age slightly lowers risk)
# p-value = 0.105 → not statistically significant
# AGE is not a strong predictor of low birth weight.

#(e) AGE vs LOW plot
# Logistic curve shows slight decrease in risk as age increases
# Relationship is weak and not statistically significant
# Age alone does not strongly explain low birth weight.

#(f) HT logistic regression
# Model: LOW ~ HT
# log(p/(1-p)) = -0.8771 + 1.2135 * HT
# HT coefficient is positive → hypertension increases risk
# p-value = 0.046 → HT is statistically significant
# Hypertension is a significant risk factor for low birth weight.