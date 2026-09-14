# Lab 4 code

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

# read data
data = pd.read_csv("SBP.txt", sep=None, engine="python")

# view first 10 rows and column names
print(data.head(10))
print(data.columns)

# convert to categorical
data["Dosage"] = data["Dosage"].astype("category")
data["Diet_Mod"] = data["Diet_Mod"].astype("category")

# boxplot by dosage and diet modification
sns.boxplot(data=data, x="Dosage", y="Change_SBP", hue="Diet_Mod")
plt.title("Change in SBP by Dosage and Diet Modification")
plt.show()

# summary stats
summary_stats = data.groupby(["Dosage", "Diet_Mod"])["Change_SBP"].agg(
    mean=lambda x: round(x.mean(),2),
    sd=lambda x: round(x.std(),2),
    n="count"
)
print(summary_stats)

# interaction plot using seaborn (replacement for statsmodels interaction_plot)
sns.pointplot(data=data, x="Dosage", y="Change_SBP", hue="Diet_Mod", dodge=True, markers=['o','s'], capsize=.1)
plt.xlabel("Dosage")
plt.ylabel("Mean Change in SBP")
plt.title("Interaction Plot: Dosage x Diet Modification")
plt.show()

# two-way ANOVA
model = smf.ols("Change_SBP ~ C(Dosage) * C(Diet_Mod)", data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
#########ANS...........

#a. the assumptions of equal variance and normality appear to be violated 
#    because the box heights vary significantly and the data is not consistently symmetric

#b. summary stats are printed by the code above

#c. strong interaction is visible in the interaction plot
####write up####

# DOSAGE effect
# H0: mean change in SBP is the same for all dosage groups.
# decision: reject H0

# DIET_MOD effect
# H0: mean change in SBP is the same for diet vs no diet.
# decision: reject H0

# INTERACTION effect
# H0: there is no interaction between Dosage and Diet_Mod on change in SBP.
# decision: reject H0 