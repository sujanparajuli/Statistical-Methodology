import pandas as pd

# read file
data = pd.read_csv("HADS_satisfaction.txt", delim_whitespace=True)

# view data
print(data.head(20))

import seaborn as sns
import matplotlib.pyplot as plt

# bar plot for anxiety groups
sns.countplot(data=data, x="ratings1", hue="anxiety")

plt.title("Patient Satisfaction by Anxiety Level")
plt.xlabel("Satisfaction Rating")
plt.ylabel("Count")
plt.show()

# kruskal wallis test for anxiety
from scipy.stats import kruskal

groups = [
    group["ratings1"].values
    for name, group in data.groupby("anxiety")
]

stat, p_value = kruskal(*groups)

print("H statistic:", stat)
print("p-value:", p_value)


# bar plot for depression groups
sns.countplot(data=data, x="ratings1", hue="depression")

plt.title("Patient Satisfaction by Depression Level")
plt.xlabel("Satisfaction Rating")
plt.ylabel("Count")
plt.show()

# kruskal wallis test for depression
groups2 = [
    group["ratings1"].values
    for name, group in data.groupby("depression")
]

stat2, p_value2 = kruskal(*groups2)

print("H statistic:", stat2)
print("p-value:", p_value2)


# answers

# a.
# a nonparametric test is required because patient satisfaction is measured
# using a likert scale (1–5), which is ordinal data. ordinal data do not meet
# the normality and interval assumptions required for parametric tests.

# b.
# the bar chart shows the distribution of patient satisfaction ratings
# for the three anxiety groups (mild, moderate, severe).

# c.
# H0: the distribution of patient satisfaction ratings is the same
# for all anxiety levels.
# Ha: at least one anxiety group has a different distribution of ratings.
# kruskal wallis results: H = 12.7742, p-value = 0.00168
# since p-value < 0.05 we reject H0.
# there is significant evidence that patient satisfaction differs
# across anxiety levels.

# d.
# yes, the bar charts support the result because the distributions
# of satisfaction ratings differ across the anxiety groups.

# e.
# the bar chart shows the distribution of patient satisfaction ratings
# for the three depression groups (mild, moderate, severe).

# f.
# H0: the distribution of patient satisfaction ratings is the same
# for all depression levels.
# Ha: at least one depression group has a different distribution.
# kruskal wallis results: H = 12.7742, p-value = 0.00168
# since p-value < 0.05 we reject H0.
# there is significant evidence that patient satisfaction differs
# across depression levels.

# g.
# yes, the bar charts support this conclusion because the ratings
# vary across the depression groups.

# h.
# both tests show that patient satisfaction differs significantly
# across anxiety levels and depression levels.
# therefore, higher levels of anxiety and depression appear to be
# associated with differences in patient satisfaction in hospital care.