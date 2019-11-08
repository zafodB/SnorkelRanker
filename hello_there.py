'''
 * Created by filip on 07/11/2019
'''

import os
import pandas as pd
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis
print("Hello there docker!")

df = pd.read_csv("/filip/json/ehealthforum/trac/training_data.txt", sep='\t',
                 names=['mdreply', 'votesh', 'votess', 'votest', 'ment', 'sameent', 'length', 'category', 'thread'])

# Keep just a bit of data for development for now
df = df.iloc[:500,]
print(df.head())
print(df['mdreply'].count())
print(df['mdreply'].count())
print(df.shape)
print(df.ndim)


# Define the label mappings for convenience
ABSTAIN = -1
NOT_RELEVANT = 0
RELEVANT = 1


@labeling_function()
def is_doctor_reply(x):
    return RELEVANT if x.mdreply == 1 else ABSTAIN


@labeling_function()
def has_votes_helpful(x):
    return RELEVANT if x.votesh > 1 else ABSTAIN


@labeling_function()
def has_votes_support(x):
    return RELEVANT if x.votess > 1 else ABSTAIN


@labeling_function()
def has_votes_thanks(x):
    return RELEVANT if x.votest > 1 else ABSTAIN


@labeling_function()
def is_long(x):
    if x.length <= 1:
        return NOT_RELEVANT
    elif 1 < x.length <= 2:
        return ABSTAIN
    elif 2 < x.length <= 10:
        return RELEVANT
    elif x.length > 10:
        return NOT_RELEVANT


lfs = [is_long, has_votes_thanks, has_votes_support, has_votes_helpful, is_doctor_reply]

applier = PandasLFApplier(lfs=lfs)

L_train = applier.apply(df=df)
# L_dev = applier.apply(df=df_dev)

print(L_train)



LFAnalysis(L=L_train, lfs=lfs).lf_summary()
# print(os.path.isdir('/filip/json/'))
#
# for root, dir, files in os.walk('/filip/json'):
#     print(root)
#     print(dir)
#     print(files)
# print(os.getcwd())
# print(os.path.abspath(os.sep))
print('sup')

