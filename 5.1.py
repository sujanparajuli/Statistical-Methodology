# regression lab - sight age distance
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

# load data
df = pd.read_csv("sightAgeDistance.txt", delim_whitespace=True)

# show first rows
print(df.head(10))

# set variables
X = df['Age']
Y = df['Distance']

# add intercept
X_with_const = sm.add_constant(X)

# fit model
model = sm.OLS(Y, X_with_const).fit()

# print regression summary
print(model.summary())

# scatterplot with regression line
plt.figure(figsize=(9, 6))
plt.scatter(X, Y, alpha=0.6)

plt.plot(X, model.predict(X_with_const),
         color='red', linestyle='--')

plt.xlabel('Age')
plt.ylabel('Distance (feet)')
plt.title('age vs distance')
plt.grid(True)
plt.show()

# predict for age 21
###This prediction is reliable because 21 is within the range of the observed data
pred_21 = model.predict([1, 21])
print("prediction at age 21:", pred_21)

# predict for age 90
###This prediction is not reliable because 90 is outside the range of the observed data, making it an extrapolation.
pred_90 = model.predict([1, 90])
print("prediction at age 90:", pred_90)

# summary statistics
summary_stats = pd.DataFrame({
    'n': [len(df['Age']), len(df['Distance'])],
    'Mean': [df['Age'].mean(), df['Distance'].mean()],
    'St Dev': [df['Age'].std(), df['Distance'].std()]
}, index=['Age', 'Distance'])

print(summary_stats.round(2))

# compute sst, sse, ssr
SSE = sum(model.resid**2)
SST = sum((Y - Y.mean())**2)
SSR = SST - SSE

print("SST:", SST)
print("SSE:", SSE)
print("SSR:", SSR)

# r squared
print("R^2:", model.rsquared)

# correlation
corr = df['Age'].corr(df['Distance'])
print("correlation:", corr)

# standard error of slope
print("sb1:", model.bse[1])

# residuals
RES = model.resid

# residual plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# residual vs age
axes[0].scatter(df['Age'], RES, alpha=0.6)
axes[0].axhline(y=0, color='red', linestyle='--')
axes[0].set_title('residual plot')

# qq plot
stats.probplot(RES, dist="norm", plot=axes[1])
axes[1].set_title('qq plot')

# histogram
axes[2].hist(RES, bins=15, edgecolor='black')
axes[2].set_title('histogram')

plt.tight_layout()
plt.show()

# residual summary
n_res = len(RES)
mean_res = RES.mean()
stdev_res = RES.std(ddof=1)

stats_RES = pd.DataFrame({
    'n': [n_res],
    'Mean': [round(mean_res, 2)],
    'St Dev': [round(stdev_res, 4)]
}, index=['RES'])

print(stats_RES)
# (c) predicted distance for age 21
# prediction at age 21: 513.54 feet
# reliable because 21 is within the range of observed ages (interpolation)

# (d) predicted distance for age 90
# prediction at age 90: 306.07 feet
# not reliable because 90 is outside the observed data (extrapolation)

# (e) summary statistics
# age: mean = 51.00, standard deviation = 21.78, n = 30
# distance: mean = 423.33, standard deviation = 81.72, n = 30

# (f) total, explained, and residual variation
# SST (total sum of squares) = 193666.67
# SSE (residual sum of squares) = 69334.02
# SSR (regression sum of squares) = 124332.64

# (g) R-squared
# R^2 = 0.642
# 64.2% of the variation in reading distance is explained by age

# (h) correlation
# correlation between age and distance = -0.801
# strong negative linear relationship: as age increases, distance decreases

# (i) sb1 (standard error of slope)
# sb1 = 0.424
# this measures uncertainty in slope estimate; smaller value indicates a precise slope

# (j) test of significance for slope
# H0: slope = 0 (no relationship)
# H1: slope != 0 (there is a relationship)
# t = -7.086, p < 0.001
# since p < 0.05, we reject H0
# conclusion: there is a significant linear relationship between age and reading distance

# (k) residual diagnostics
# residual plot shows random scatter around zero:linear model appropriate
# Q-Q plot: points approximately along straight line :residuals roughly normal
# histogram: rnot symmetric
# residual standard deviation = 48.8961
# overall: linear regression assumptions reasonably satisfied