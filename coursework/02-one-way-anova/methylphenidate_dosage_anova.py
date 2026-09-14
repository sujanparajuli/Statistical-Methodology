import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load the data
mp_data = pd.read_csv("methylphenidate_ANOVA.txt", sep=r"\s+", header=0)
print(mp_data.head())

# Separate the data by dosage
D0 = mp_data.loc[mp_data["dosage"]=="D0", "correct.responses"]
D15 = mp_data.loc[mp_data["dosage"]=="D15", "correct.responses"]
D30 = mp_data.loc[mp_data["dosage"]=="D30", "correct.responses"]
D60 = mp_data.loc[mp_data["dosage"]=="D60", "correct.responses"]

# Descriptive statistics
for dose, data in zip(["D0","D15","D30","D60"], [D0,D15,D30,D60]):
    print(f"{dose} - Mean: {data.mean():.2f}, SD: {data.std(ddof=1):.2f}, Median: {data.median()}, n: {data.count()}")

# Boxplot
mp_data.boxplot(column="correct.responses", by="dosage")
plt.suptitle("")
plt.title("DOG Task Performance by Dosage")
plt.ylabel("Number of Correct Responses")
plt.show()

# Histograms and Q-Q plots
plt.figure(figsize=(10,6))
plt.subplot(2,4,1); plt.hist(D0); plt.title("D0")
plt.subplot(2,4,2); plt.hist(D15); plt.title("D15")
plt.subplot(2,4,3); plt.hist(D30); plt.title("D30")
plt.subplot(2,4,4); plt.hist(D60); plt.title("D60")
plt.subplot(2,4,5); stats.probplot(D0, plot=plt)
plt.subplot(2,4,6); stats.probplot(D15, plot=plt)
plt.subplot(2,4,7); stats.probplot(D30, plot=plt)
plt.subplot(2,4,8); stats.probplot(D60, plot=plt)
plt.tight_layout()
plt.show()

# Run ANOVA
F_stat, p_value = stats.f_oneway(D0, D15, D30, D60)
print("ANOVA F-statistic:", F_stat)
print("p-value:", p_value)

#Formal
# H0: μ_D0 = μ_D15 = μ_D30 = μ_D60
# H1: At least one dosage mean is different

# α = 0.05
# F = 1.4815
# p-value = 0.2247
# Since p-value > α, fail to reject H0
# There is no strong evidence that the dosage affects DOG task performance
# Post-hoc t-tests are not necessary because ANOVA was not significant

#QQ: points roughly follow the diagonal line, so can say normality is met.
#boxplot: No clear difference in mean performance between dosages,showing a non-significant(small) F-statistic.

