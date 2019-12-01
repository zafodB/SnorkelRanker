'''
 * Created by filip on 26/11/2019
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_valid = pd.read_csv("/filip/json/ehealthforum/trac/validation_df.txt", sep="\t", header=0)

print(list(df_valid.columns))

    df_valid['predicted_prob'].plot.hist(bins=10)
plt.savefig("/filip/json/ehealthforum/trac/plot.png")

print(df_valid.head())
