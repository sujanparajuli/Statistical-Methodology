# Lab 1 Problem #1

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

# Input data
YF = np.array([29, 34, 33, 27, 28, 32, 31, 34, 32, 27])
OF = np.array([18, 15, 23, 13, 12])

# Adjust YF for testing difference > 10
YF_adjusted = YF - 10

# Two-sample t-test (Welch's t-test)
t_stat, p_value_two_sided = stats.ttest_ind(YF_adjusted, OF, equal_var=False)

# One-sided p-value
if t_stat > 0:
    p_value = p_value_two_sided / 2
else:
    p_value = 1 - p_value_two_sided / 2

print("t statistic:", t_stat)
print("one-sided p-value:", p_value)

# Boxplot
namevec = ["YF"] * len(YF)
namevec1 = ["OF"] * len(OF)

my_data = pd.DataFrame({
    "group": namevec + namevec1,
    "lean_angle": list(YF) + list(OF)
})

my_data.boxplot(column="lean_angle", by="group")
plt.suptitle("")
plt.title("Maximum Lean Angle by Group")
plt.xlabel("Group")
plt.ylabel("Lean Angle")
plt.show()

# Normal Q-Q plot: YF
(osm, osr), (slope, intercept, r) = stats.probplot(YF)

plt.figure()
plt.scatter(osm, osr)
x_line = np.array([osm.min(), osm.max()])
y_line = slope * x_line + intercept
plt.plot(x_line, y_line)
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Data")
plt.title("Normal Q-Q Plot: YF")
plt.show()

# Normal Q-Q plot: OF
(osm, osr), (slope, intercept, r) = stats.probplot(OF)

plt.figure()
plt.scatter(osm, osr)
x_line = np.array([osm.min(), osm.max()])
y_line = slope * x_line + intercept
plt.plot(x_line, y_line)
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Data")
plt.title("Normal Q-Q Plot: OF")
plt.show()
#############FORMAL WRITE_UP###################
#Ho: The true average maximum lean angle for older females is 10 degrees smaller or less than that for younger females.
#Ha: The true average maximum lean angle for older females is more than 10 degrees smaller than that for younger females.

#t = 2.076
#p-value = 0.043

#Reject Ho, there is sufficient evidence to show that the true average maximum lean angle for older females is more than 10 degrees smaller than that for younger females.
#The box plot suggests different mean and the normal plot is also satisfied with the result.
