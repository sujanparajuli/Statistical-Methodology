# Lab 4 code

import pandas as pd

# read data
data = pd.read_csv("exercise.txt", sep=None, engine="python")

# view first 10 rows and column names
print(data.head(10))
print(data.columns)

import seaborn as sns
import matplotlib.pyplot as plt

# convert to categorical
data["exercise_status"] = data["exercise_status"].astype("category")
data["gender"] = data["gender"].astype("category")

# boxplot by exercise status and gender
sns.boxplot(data=data, x="exercise_status", y="hr", hue="gender")
plt.title("Heart Rate Change by Exercise Status and Gender")
plt.show()

# summary stats
summary_stats = data.groupby(["exercise_status", "gender"])["hr"].agg(
    mean=lambda x: round(x.mean(),2),
    sd=lambda x: round(x.std(),2),
    n="count"
)
print(summary_stats)

# interaction plot using seaborn
sns.pointplot(data=data, x="exercise_status", y="hr", hue="gender", dodge=True, markers=['o','s'], capsize=.1)
plt.xlabel("Exercise Status")
plt.ylabel("Mean Heart Rate Change")
plt.title("Interaction Plot: Exercise Status x Gender")
plt.show()

import statsmodels.api as sm
import statsmodels.formula.api as smf

# two-way ANOVA
model = smf.ols("hr ~ C(exercise_status) * C(gender)", data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
#########ANS...........

#a. the assumptions of equal variance and normality appear to be violated 
#    because the box heights vary and the data is not completely symmetric

#b. summary stats are printed by the code above

#c. strong interaction is visible in the interaction plot
####write up####

# EXERCISE_STATUS effect
# H0: mean heart rate change is the same for runners and sedentary group.
# decision: reject H0 

# GENDER effect
# H0: mean heart rate change is the same for males and females.
# decision: reject H0 

# INTERACTION effect
# H0: there is no interaction between exercise_status and gender on heart rate.
# decision:reject H0 