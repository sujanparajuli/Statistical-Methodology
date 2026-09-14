import pandas as pd

# load the dataset
data = pd.read_csv("orings.txt", sep="\s+")

# preview data
print("data preview:")
print(data.head())

# column names
print("\ncolumn names:")
print(data.columns.tolist())

# basic statistics
print("\nbasic statistics:")
print(data.describe())

import matplotlib.pyplot as plt

# scatter plot of temperature vs failure
plt.scatter(data['temp'], data['failure'],
            alpha=0.7,
            edgecolor='navy',
            color='lightblue',
            s=60)

plt.xlabel('temperature')
plt.ylabel('failure (0/1)')
plt.title('scatterplot: temperature vs failure')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

import statsmodels.formula.api as smf

# fit logistic regression model
model = smf.logit('failure ~ temp', data=data).fit()

# model summary
print(model.summary())

# aic value
print("AIC =", model.aic)

import numpy as np

# scatter plot with logistic curve
plt.scatter(data['temp'], data['failure'],
            alpha=0.7,
            edgecolor='navy',
            color='lightblue',
            s=60,
            label='observed data (0 or 1)')

# temperature range for smooth curve
temp_range = np.linspace(data['temp'].min(), data['temp'].max(), 200)

# dataframe for prediction
pred_df = pd.DataFrame({'temp': temp_range})

# predicted probabilities
predicted_probs = model.predict(pred_df)

# logistic curve
plt.plot(temp_range, predicted_probs,
         color='red',
         linewidth=2.8,
         label='fitted logistic curve')

plt.xlabel('temperature')
plt.ylabel('probability of failure')
plt.title('logistic regression: failure ~ temperature')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# regression equation
params = model.params
print("prediction equation:")
print("log(p/(1-p)) =", params[0], "+", params[1], "* temp")


# a) prediction equation
# log(p/(1-p)) = 10.8753 - 0.1713 * temp
# p = probability of failure

# b) plot
# as temperature increases, failure probability decreases (logistic curve goes down)

# c) b1 (temp coefficient) = -0.1713
# negative means: higher temperature -> lower chance of O-ring failure

# d) 95% CI for odds ratio
# [0.715, 0.992]
# since interval is below 1, temperature significantly reduces odds of failure

# e) probability at 31°F = 0.996
# very high predicted failure probability
# yes, this is extrapolation because 31°F is outside observed range (53 to 81)

# f) at temp = 70
# odds = 0.327
# log(odds) = -1.117

# g) probabilities
# at 53°F = 0.858 (high failure chance)
# at 81°F = 0.047 (low failure chance)