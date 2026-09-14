# regression lab - old faithful eruptions
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

# load data
df = pd.read_csv("OldFaithful.txt", delim_whitespace=True)  # sep='\s+' also works

# show first 10 rows
print(df.head(10))

# set variables
X = df['Duration']    # duration of eruption (minutes)
Y = df['Time']        # time until next eruption (minutes)

# add intercept
X_with_const = sm.add_constant(X)

# fit linear regression model
model = sm.OLS(Y, X_with_const).fit()

# print regression summary
print(model.summary())

# (a) scatterplot with regression line
plt.figure(figsize=(9, 6))
plt.scatter(X, Y, alpha=0.6)
plt.plot(X, model.predict(X_with_const), color='red', linestyle='--')
plt.xlabel('duration (minutes)')
plt.ylabel('time until next eruption (minutes)')
plt.title('old faithful duration vs time until next eruption')
plt.grid(True)
plt.show()

# (b) regression equation
b0 = model.params[0]
b1 = model.params[1]
print("regression equation: next eruption time =", round(b0,2), "+", round(b1,2), "* duration")

# (c) predict next eruption for duration = 10 minutes
pred_10 = model.predict([1, 10])
print("prediction for duration 10:", pred_10)
# reliable if 10 is within observed duration range (interpolation)

# (d) summary statistics
summary_stats = pd.DataFrame({
    'n': [len(df['Duration']), len(df['Time'])],
    'Mean': [df['Duration'].mean(), df['Time'].mean()],
    'St Dev': [df['Duration'].std(), df['Time'].std()]
}, index=['Duration', 'NextEruptionTime'])
print(summary_stats.round(2))

# (e) compute SST, SSE, SSR
SSE = sum(model.resid**2)
SST = sum((Y - Y.mean())**2)
SSR = SST - SSE
print("SST:", SST)
print("SSE:", SSE)
print("SSR:", SSR)

# (f) R-squared
print("R^2:", model.rsquared)
# shows % variation in next eruption time explained by duration

# (g) correlation
corr = df['Duration'].corr(df['Time'])
print("correlation:", corr)

# (h) sb1 (standard error of slope)
print("sb1:", model.bse[1])
# sb1 measures precision of slope estimate

# (i) residual summary
RES = model.resid
n_res = len(RES)
mean_res = RES.mean()
stdev_res = RES.std(ddof=1)
stats_RES = pd.DataFrame({
    'n': [n_res],
    'Mean': [round(mean_res, 2)],
    'St Dev': [round(stdev_res, 4)]
}, index=['RES'])
print(stats_RES)


# (k) residual diagnostics
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].scatter(df['Duration'], RES, alpha=0.6)
axes[0].axhline(y=0, color='red', linestyle='--')
axes[0].set_title('residual plot')
stats.probplot(RES, dist="norm", plot=axes[1])
axes[1].set_title('qq plot')
axes[2].hist(RES, bins=15, edgecolor='black')
axes[2].set_title('histogram')
plt.tight_layout()
plt.show()
# (b) regression equation
# next eruption time = 35.30 + 11.82 * duration

# (c) prediction for duration = 10 minutes
# predicted time = 153.55 minutes
# not reliable because 10 minutes is far outside the observed range (extrapolation)

# (d) summary statistics
# duration: n = 35, mean = 3.32 minutes, standard deviation = 1.09 minutes
# next eruption time: n = 35, mean = 74.51 minutes, standard deviation = 13.23 minutes

# (e) sum of squares
# SST (total variation) = 5954.74
# SSE (unexplained variation) = 354.39
# SSR (explained variation) = 5600.36

# (f) R-squared
# R^2 = 0.9405
# 94.05% of the variation in next eruption time is explained by the duration

# (g) correlation
# correlation = 0.970
# very strong positive linear relationship: longer duration → longer time until next eruption

# (h) sb1 (standard error of slope)
# sb1 = 0.518
# measures precision of the slope estimate; smaller value → more precise estimate

# (i) residual summary
# number of residuals = 35
# mean of residuals = 0.0
# standard deviation of residuals = 3.2285

# (j) test of significance for slope
# H0: slope = 0 (no relationship)
# H1: slope != 0 (relationship exists)
# t = 22.836, p < 0.001
# since p < 0.05, reject H0
# conclusion: there is a significant relationship between duration and next eruption time

# (k) residual diagnostics
# residual plot: points scattered randomly around zero  linear model appropriate
# Q-Q plot: points approximately on straight line →residuals roughly normal
