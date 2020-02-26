#!/usr/bin/env python

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


# Load the dataset with pandas
df = pd.read_hdf()
# Columns:
#  - graph (array)
#  - subject (categorical)
#  - session (categorical)
#  - pipeline (categorical)
#  - seed (categorical)
#  - directions (int)
#  - mca (categorical)

# Set the linear model equation
model_equation = "graph ~ subject + session + pipeline + seed + directions"

# Filter dataframe to only have a single instance per individual. Simply: pick
# the one corresponding to MCA sim numero uno
df_nomca = df[df["mca"] == "1"]
model_nomca = ols(model_equation, data=df_nomca)
aovtable_nomca = sm.stats.anova_lm(model_nomca, typ=2)
print(aovtable_nomca)

# Re-perform the experiment with the MCA simulations included
model_mca = ols(model_equation, data=df)
aovtable_mca = sm.stats.anova_lm(model_mca, typ=2)
print(aovtable_mca)