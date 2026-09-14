import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

################################
# Question 3: Relationship between X and Y

# Data
X = np.array([-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
Y = np.array([1, 14, 25, 34, 41, 46, 49, 50, 49, 46, 41, 34, 25, 14, 1])

# Correlation
corr_XY = np.corrcoef(X, Y)[0, 1]
print(f"Correlation between X and Y: {corr_XY:.2f}")

# Linear regression
X_const = sm.add_constant(X)
model = sm.OLS(Y, X_const).fit()
print(model.summary())

# Scatterplot with regression line
plt.figure(figsize=(6,4))
plt.scatter(X, Y, color='blue', label='Data points')
plt.plot(X, model.predict(X_const), color='red', label='Fitted line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Question 3: Scatterplot with LSR Line')
plt.legend()
plt.show()

# Comment:
# The correlation is close to 0, but the scatterplot shows a clear non-linear relationship.
# This demonstrates that correlation alone can be misleading when the relationship is non-linear.

################################
# Question 4: Anscombe's Quartet

# Data
anscombe_data = {
    'X1': [10,8,13,9,11,14,6,4,12,7,5],
    'Y1': [8.04,6.95,7.58,8.81,8.33,9.96,7.24,4.26,10.84,4.82,5.68],
    'X2': [10,8,13,9,11,14,6,4,12,7,5],
    'Y2': [9.14,8.14,8.74,8.77,9.26,8.10,6.13,3.10,9.13,7.26,4.74],
    'X3': [10,8,13,9,11,14,6,4,12,7,5],
    'Y3': [7.46,6.77,12.74,7.11,7.81,8.84,6.08,5.39,8.15,6.42,5.73],
    'X4': [8,8,8,8,8,8,8,19,8,8,8],
    'Y4': [6.58,5.76,7.71,8.84,8.47,7.04,5.25,12.50,5.56,7.91,6.89]
}

df = pd.DataFrame(anscombe_data)

# Summary statistics
summary_stats = pd.DataFrame({
    'Mean': df.mean(),
    'Std Dev': df.std(ddof=1)
}).round(2)
print("\nSummary Statistics for Anscombe's Quartet:")
print(summary_stats)

# Correlations
correlations = df[['X1','Y1','X2','Y2','X3','Y3','X4','Y4']].corr().iloc[::2,1::2]
correlations = correlations.rename(columns={'Y1':'X1 vs Y1','Y2':'X2 vs Y2','Y3':'X3 vs Y3','Y4':'X4 vs Y4'})
print("\nCorrelations:")
print(correlations.round(2))

# Scatterplots with LSR lines
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

datasets = [('X1','Y1'), ('X2','Y2'), ('X3','Y3'), ('X4','Y4')]
titles = ['Dataset 1', 'Dataset 2', 'Dataset 3', 'Dataset 4']

for ax, (x_col, y_col), title in zip(axs.flatten(), datasets, titles):
    X_vals = df[x_col]
    Y_vals = df[y_col]
    X_const = sm.add_constant(X_vals)
    model = sm.OLS(Y_vals, X_const).fit()
    
    ax.scatter(X_vals, Y_vals, color='blue')
    ax.plot(X_vals, model.predict(X_const), color='red')
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

plt.tight_layout()
plt.show()

# Comments:
# All four datasets have nearly identical means, standard deviations, and correlations.
# However, the scatterplots reveal very different relationships:
# - Dataset 1: Linear trend
# - Dataset 2: Linear trend with slight curvature
# - Dataset 3: Outlier affects the slope
# - Dataset 4: Single outlier dominates pattern
# This illustrates why plotting data is crucial before relying on summary statistics alone.