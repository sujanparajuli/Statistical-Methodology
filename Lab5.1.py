import pandas as pd

# read file
data = pd.read_csv("RTP.txt", delim_whitespace=True)

# view data
print(data.head(20))

import matplotlib.pyplot as plt

# split by gender
male_RTP = data.loc[data["gender"] == "male", "RTP"]
female_RTP = data.loc[data["gender"] == "female", "RTP"]

# histograms
fig, axes = plt.subplots(2,1)

axes[0].hist(male_RTP)
axes[0].set_title("Males RTP")

axes[1].hist(female_RTP)
axes[1].set_title("Females RTP")

plt.tight_layout()
plt.show()

# mann whitney test (wilcoxon rank sum)
from scipy.stats import mannwhitneyu

stat, p_value = mannwhitneyu(
    female_RTP,
    male_RTP,
    alternative="two-sided"
)

print("U statistic:", stat)
print("p-value:", p_value)


# parametric test (two sample t-test)
from scipy.stats import ttest_ind

t_stat, p_val = ttest_ind(
    female_RTP,
    male_RTP,
    equal_var=False
)

print("t statistic:", t_stat)
print("p-value:", p_val)
####annssss###
#b. the histograms show the rtp times are slightly right skewed.
#c. a nonparametric test is required because the rtp data are not normally distributed
# and the variable is discrete (days). nonparametric tests do not assume normality.
#d.the appropriate test is the wilcoxon rank sum test (mann whitney test).

# e.
# H0: the distribution of rtp days is the same for males and females.
# Ha: the distribution of rtp days differs between males and females.
# mann whitney test results: U = 32468.0, p-value = 0.28697
# since p-value > 0.05 we fail to reject H0.
# there is no significant difference in rtp time between male and female players.

# hypothesis summary (for write up)
# H0: 
# distribution of RTP days is the same for males and females
# Ha: distribution of RTP days differs between males and females
# if p-value < 0.05 reject H0, otherwise fail to reject

# f.yes, the plots support the conclusion because the histograms for males and females
# look fairly similar and show no large shift between the two distributions.

# g.the analogous parametric test is the two sample t-test.
# t statistic = 0.62137, p-value = 0.53469
# since p-value > 0.05 we fail to reject H0.

# h. the general conclusion did not change.
# both the mann whitney test and the two sample t-test indicate that
# there is no statistically significant difference in rtp time between males and females.