#Lab 1 problem 2
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Paired fabric data
unabraded = np.array([36.4, 55.0, 51.5, 38.7, 43.2, 48.8, 25.6, 49.8])
abraded   = np.array([28.5, 20.0, 46.0, 34.5, 36.5, 52.5, 26.5, 46.5])

# Differences (U - A)
diff = unabraded - abraded

# Paired t-test (one-sided)
t_stat, p_two_sided = stats.ttest_rel(unabraded, abraded)
p_one_sided = p_two_sided / 2  # because we are testing 'greater'

print("t statistic:", t_stat)
print("one-sided p-value:", p_one_sided)

# Boxplot of both samples
plt.figure()
plt.boxplot([unabraded, abraded], labels=["Unabraded", "Abraded"])
plt.ylabel("Breaking Load (kg/25 mm)")
plt.title("Breaking Load by Fabric Condition")
plt.show()

# Normal probability plot (Q-Q plot) for each sample
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
stats.probplot(unabraded, dist="norm", plot=plt)
plt.title("Normal Q-Q Plot: Unabraded")

plt.subplot(1,2,2)
stats.probplot(abraded, dist="norm", plot=plt)
plt.title("Normal Q-Q Plot: Abraded")

plt.tight_layout()
plt.show()
################Formal Write-UP############
#Ho: The true mean breaking load of unabraded fabrics does not differ from abraded fabrics.
#Ha: The true mean breaking load of unabraded fabrics is greater than abraded fabrics.

#t = 1.728607
#one-sided p-value = 0.063754

#Fail to reject Ho, there is not sufficient evidence at the 0.01 significance level to show that the true mean breaking load of unabraded fabrics is greater than abraded fabrics.

#Both Q-Q plots show points lying roughly along the straight line with no strong curvature or outliers, indicating that the unabraded and abraded breaking load data are approximately normally distributed.