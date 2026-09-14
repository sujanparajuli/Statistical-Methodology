import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

chol_data = pd.read_csv("cholesterol_ANOVA.txt", sep=r"\s+", header=0)
print(chol_data.head())

mg_dL = chol_data["mg_dL"]

print("Mean:", mg_dL.mean())
print("SD:", mg_dL.std(ddof=1))
print("Median:", mg_dL.median())
print("Sample size:", mg_dL.count())

# Form a data frame to group by treatment
group_stats = chol_data.groupby("Treatment")["mg_dL"].agg(
    n="count",
    mean=lambda x: round(x.mean(), 1),
    std_dev=lambda x: round(x.std(ddof=1), 2)
)

print(group_stats)

# This gives us the five number summary for each treatment
print(chol_data.groupby("Treatment")["mg_dL"].describe())

# Plot - run all the below together
chol_data.boxplot(column="mg_dL", by="Treatment")
plt.suptitle("")
plt.title("Cholesterol by Treatment")
plt.ylabel("mg/dL")
plt.show()

# Do the boxplots indicate a difference in mean mg_dL?

# Isolate the mg_dL data for each treatment
CONTROL = chol_data.loc[chol_data["Treatment"] == "control", "mg_dL"]
DAY2 = chol_data.loc[chol_data["Treatment"] == "02_day", "mg_dL"]
DAY4 = chol_data.loc[chol_data["Treatment"] == "04_day", "mg_dL"]
DAY14 = chol_data.loc[chol_data["Treatment"] == "14_Day", "mg_dL"]

# Generate the histogram and qq plots (run entire block below)
plt.figure(figsize=(10,6))
plt.subplot(2,4,1); plt.hist(CONTROL); plt.title("CONTROL")
plt.subplot(2,4,2); plt.hist(DAY2); plt.title("02_DAY")
plt.subplot(2,4,3); plt.hist(DAY4); plt.title("04_DAY")
plt.subplot(2,4,4); plt.hist(DAY14); plt.title("14_DAY")
plt.subplot(2,4,5); stats.probplot(CONTROL, plot=plt)
plt.subplot(2,4,6); stats.probplot(DAY2, plot=plt)
plt.subplot(2,4,7); stats.probplot(DAY4, plot=plt)
plt.subplot(2,4,8); stats.probplot(DAY14, plot=plt)
plt.tight_layout()
plt.show()

# Run ANOVA
F_stat, p_value = stats.f_oneway(CONTROL, DAY2, DAY4, DAY14)

print("ANOVA F-statistic:", F_stat)
print("p-value:", p_value)
# Since we had a significant result: Run side by side comparisons (Two means) between treatment days
# 02_DAY vs 04_DAY
t1 = stats.ttest_ind(DAY2, DAY4, equal_var=True)
# 02_DAY vs 14_DAY
t2 = stats.ttest_ind(DAY2, DAY14, equal_var=True)
# 04_DAY vs 14_DAY (two-sided)
t3 = stats.ttest_ind(DAY4, DAY14, equal_var=True)

print("02_DAY vs 04_DAY:", t1) #not significant
print("02_DAY vs 14_DAY:", t2) #signimicant
print("04_DAY vs 14_DAY:", t3) #significant

#________________________----------------------------
#Questions and answers
#1. They are not all very close, 4day is much higher, and 14day is much lower.

##2. Spread between group means suggests that the F-statistic will be large, 
#    because ANOVA measures the ratio of between-group variance to within-group variance. 
#    The bigger the differences in group means relative to their spreads, the larger the F-statistic.

###3.The assumption of normality is reasonably met for all groups also mentioning 14day lil bit off.

####4#
# H0: μ_control = μ_02_day = μ_04_day = μ_14_day (all group means are equal)
# H1: At least one group mean is different

# α = 0.05
# F =13.9188
# p-value = 1.069e-07
# Since p-value < α, reject H0
# There is strong evidence that at least one group mean is different

#####5
# Post-hoc two-sample t-tests
# Test 1: 02_DAY vs 04_DAY
# H0: μ_02_day = μ_04_day
# H1: μ_02_day =! μ_04_day
# α = 0.05
# t = -0.4004
# p-value = 0.691
# Since p-value > α, fail to reject H0
# No significant difference between 02_DAY and 04_DAY

# Test 2: 02_DAY vs 14_DAY
# H0: μ_02_day = μ_14_day
# H1: μ_02_day =! μ_14_day
# α = 0.05
# t = 3.8264
# p-value = 0.00033
# Since p-value < α, reject H0
#Significant difference between 02_DAY and 14_DAY

# Test 3: 04_DAY vs 14_DAY
# H0: μ_04_day = μ_14_day
# H1: μ_04_day +1 μ_14_day
# α = 0.05
# t = 3.1288
# p-value = 0.00293
# Since p-value < α, reject H0
# Significant difference between 04_DAY and 14_DAY



