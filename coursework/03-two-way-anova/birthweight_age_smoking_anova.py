# Lab 4 code

import pandas as pd

data = pd.read_csv("birthweight.txt", sep=None, engine="python")  
# sep=None lets pandas auto-detect whitespace or tabs

# View first 10 rows
print(data.head(10))

# See column names
print(data.columns)

import seaborn as sns
import matplotlib.pyplot as plt

# Create a boxplot for each age category, paired by smoking status
sns.boxplot(
    data=data,
    x="Age",
    y="Birthweight",
    hue="Smoke_Status"
)
plt.title("Birthweight by Age and Smoke Status")
plt.show()

# Compute summary stats for each age combo of age group and smoking status
summary_stats = (
    data
    .groupby(["Age", "Smoke_Status"])["Birthweight"]
    .agg(
        mean=lambda x: round(x.mean(), 2),
        sd=lambda x: round(x.std(), 2),
        n="count"
    )
)

print(summary_stats)

# Import this for interaction plot
from statsmodels.graphics.factorplots import interaction_plot

# Create interaction plot with legend, labels and title
fig = interaction_plot(
    x=data["Age"],
    trace=data["Smoke_Status"],
    response=data["Birthweight"]
)
plt.xlabel("Age")
plt.ylabel("Mean Birthweight")
plt.show()

# Finally, run the two factor ANOVA
# For me to run the below import commands, I had to do the following: 
    # Type this into the console on the right: conda install statsmodels
    # Restart the kernel and run everything above again
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Fit the two-way ANOVA model
model = smf.ols(
    "Birthweight ~ C(Age) * C(Smoke_Status)",
    data=data
).fit()

# In the above, C() is important. For example C(Age) tells python that Age is categorical

anova_table = sm.stats.anova_lm(model, typ=2)

print(anova_table)

##########ANS...........
#a. the assumptions of equal variance and normality appear to be violated because the box heights vary significantly and the data is not consistently symmetric

#b. Summary stats in code
#c. No and it seems somehow parallel
###########ANOVA Test-----
# Null Hypotheses for the two-factor ANOVA
# H0 (Age): Mean birthweight is the same for all age groups.
# H0 (Smoke_Status): Mean birthweight is the same for smokers and non-smokers.
# H0 (Interaction): There is no interaction between Age and Smoke_Status on Birthweight.
## for age and smoke status we reject the null hypothesis but fail to reject the interaction