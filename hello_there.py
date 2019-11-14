'''
 * Created by filip on 07/11/2019
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis, LabelModel

print("Hello there docker!")

# df = pd.read_csv("/filip/json/ehealthforum/trac/training_data.txt", sep='\t',
#                  names=['mdreply', 'votesh', 'votess', 'votest', 'ment', 'sameent', 'length', 'category', 'thread'])

# df = pd.read_csv("/filip/json/ehealthforum/trac/training_data_snorkel5k.txt", sep='\t', header=0, error_bad_lines=False)
df = pd.read_csv("/filip/json/ehealthforum/trac/training_data_snorkel_10k_full.txt", sep='\t', header=0, error_bad_lines=False)


def split_by_semicolon(input_text):
    if type(input_text) is str:
        return input_text.split(sep=';')
    else:
        return ""


df['document_annotations'] = df['document_annotations'].apply(split_by_semicolon)
df['query_annotations'] = df['query_annotations'].apply(split_by_semicolon)

# print(df['document_annotations'].head())

# print(list(df.columns))
# Keep just a bit of data for development for now
# df = df.iloc[:500,]
# print(df.head())
# print()
# print(df.info())
# print(df.describe())
# print()
# print(df.shape)
# print(df.ndim)


# Define the label mappings for convenience
ABSTAIN = -1
NOT_RELEVANT = 0
RELEVANT = 1

# ABSTAIN = -1
# NOT_RELEVANT = 0
# WEAK = 1
# MEDIUM = 2
# STRONG = 3


@labeling_function()
def is_doctor_reply(x):
    return RELEVANT if x.document_is_doctor_reply or x.document_user_status == "Experienced User" else NOT_RELEVANT


@labeling_function()
def has_votes(x):
    total_votes = int(x.document_number_votes_h) + int(x.document_number_votes_s) + int(x.document_number_votes_t)
    return RELEVANT if total_votes >= 1 else NOT_RELEVANT


@labeling_function()
def is_long(x):
    text_length = len(x.document_text)
    return RELEVANT if text_length > 300 else NOT_RELEVANT


@labeling_function()
def is_same_thread(x):
    if x.document_thread == x.query_thread:
        return RELEVANT
    elif x.document_thread != x.query_thread and x.document_category == x.query_category:
        return RELEVANT
    else:
        return NOT_RELEVANT


@labeling_function()
def enity_overlap(x):
    return RELEVANT if len(set(x.document_annotations).intersection(set(x.query_annotations))) > 0 else NOT_RELEVANT


@labeling_function()
def has_entities(x):
    return RELEVANT if len(set(x.document_annotations)) > 1 else NOT_RELEVANT


@labeling_function()
def has_entities(x):
    return RELEVANT if len(set(x.document_annotations)) > 1 else NOT_RELEVANT


'''
['query_category', 'query_thread', 'query_text', 'query_annotations', 'typ_dsyn', 'typ_patf', 'typ_sosy', 'typ_dora', 
'typ_fndg', 'typ_menp', 'typ_chem', 'typ_orch', 'typ_horm', 'typ_phsu', 'typ_medd', 'typ_bhvr', 'typ_diap', 'typ_bacs',
 'typ_enzy', 'typ_inpo', 'typ_elii', 'document_category', 'document_thread', 'document_text', 'document_is_doctor_reply',
  'document_number_votes_h', 'document_number_votes_s', 'document_number_votes_t', 'document_user_status', 
  'document_annotations', 'd_typ_dsyn', 'd_typ_patf', 'd_typ_sosy', 'd_typ_dora', 'd_typ_fndg', 'd_typ_menp', 
  'd_typ_chem', 'd_typ_orch', 'd_typ_horm', 'd_typ_phsu', 'd_typ_medd', 'd_typ_bhvr', 'd_typ_diap', 'd_typ_bacs', 
  'd_typ_enzy', 'd_typ_inpo', 'd_typ_elii', 'bm25_relevant', 'bm25_score']
'''

# print(df.document_user_status.unique())
df_train, df_valid = train_test_split(df, test_size=0.1, random_state=1234556, stratify=df.bm25_relevant)
Y_valid = df_valid.bm25_relevant.values
Y_train = df_train.bm25_relevant.values

lfs = [is_long, has_votes, is_doctor_reply, is_same_thread, has_entities, enity_overlap]
# lfs = [has_votes, is_doctor_reply, is_same_thread, enity_overlap]
# lfs = [is_same_thread, enity_overlap, is_doctor_reply]

applier = PandasLFApplier(lfs=lfs)

# print(df_train.loc[23467])
# print(df_train.sample(5))

#
# L_dev = applier.apply(df=df_dev)
L_train = applier.apply(df=df_train)
L_valid = applier.apply(df=df_valid)
#
print(L_train)
# #
print(LFAnalysis(L=L_train, lfs=lfs).lf_summary(Y=Y_train))


label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=20000, lr=0.0001, log_freq=10, seed=81794)

label_model_acc = label_model.score(L=L_valid, Y=Y_valid)["accuracy"]
print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")
print(label_model.get_weights())

print('\nsup')


# TODO  check this later
def plot_label_frequency(L):
    plt.hist((L != ABSTAIN).sum(axis=1), density=True, bins=range(L.shape[1]))
    plt.xlabel("Number of labels")
    plt.ylabel("Fraction of dataset")
    plt.savefig("/filip/json/ehealthforum/trac/plot.png")


# plot_label_frequency(L_train)
