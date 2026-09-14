# Lab code

import pandas as pd

# Read the data - make sure your workspace is set to where this file is saved
data = pd.read_csv("HR.txt", sep=None, engine="python")  
# sep=None lets pandas auto-detect whitespace or tabs

# View first 10 rows
print(data.head(10))

# See column names
print(data.columns)

import seaborn as sns
import matplotlib.pyplot as plt

# Create a boxplot for each age category, paired by gender
sns.boxplot(
    data=data,
    x="AGE",
    y="HR",
    hue="GENDER"
)
plt.title("Heart Rate by Age and Gender")
plt.show()

# Compute summary stats for each age combo of age group and gender
summary_stats = (
    data
    .groupby(["AGE", "GENDER"])["HR"]
    .agg(
        mean=lambda x: round(x.mean(), 2),
        sd=lambda x: round(x.std(), 2),
        n="count"
    )
)

print(summary_stats)

# Import this for interaction plot
from statsmodels.graphics.factorplots import interaction_plot

# Create interaction plot
fig = interaction_plot(
    x=data["AGE"],
    trace=data["GENDER"],
    response=data["HR"]
)
plt.xlabel("Age Group")
plt.ylabel("Mean Heart Rate")
plt.show()

# Finally, run the two factor ANOVA
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Fit the two-way ANOVA model
model = smf.ols(
    "HR ~ C(AGE) * C(GENDER)",
    data=data
).fit()

# C() tells python the variables are categorical

anova_table = sm.stats.anova_lm(model, typ=2)

print(anova_table)
##########ANS...........
#a. the assumptions of equal variance and normality appear to be violated because the box heights vary significantly and the data is not consistently symmetric

#b. Summary stats in code
#c. No and it seems somehow parallel
####write up####
# AGE effect
# H0: mean heart rate is the same for all age groups.
# decision: reject H0

# GENDER effect
# H0: mean heart rate is the same for males and females.
# decision: reject H0 if p-value 

# INTERACTION effect
# H0: there is no interaction between AGE and GENDER on heart rate.
# decision: fail to reject H0 if p-value