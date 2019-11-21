'''
 * Created by filip on 07/11/2019
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, average_precision_score
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis, LabelModel

print("Hello there docker!")

METRIC_AT = 5
PROBABILITY_CUTOFF = 0.9

# df = pd.read_csv("/filip/json/ehealthforum/trac/training_data.txt", sep='\t',
#                  names=['mdreply', 'votesh', 'votess', 'votest', 'ment', 'sameent', 'length', 'category', 'thread'])

df = pd.read_csv("/filip/json/ehealthforum/trac/training_data_snorkel_10k_titles.txt", sep='\t', header=0,
                 error_bad_lines=False, encoding="ISO-8859–1")


def split_by_char(input_text, character=';'):
    try:
        return input_text.split(sep=character)
    except AttributeError:
        return ""


df['document_annotations'] = df['document_annotations'].apply(split_by_char)
df['query_annotations'] = df['query_annotations'].apply(split_by_char)
df.relationships_list = df.relationships_list.apply(split_by_char, args=(',',))

# print(df['document_annotations'].head())

# print(list(df.columns))
# Keep just a bit of data for development for now
# df = df.iloc[:500,]
print(df.relationships_list.head())
# print()
print(df.info())
print(df.describe())
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
def has_type_dsyn(x):
    return RELEVANT if x.d_typ_dsyn > 0 else ABSTAIN


@labeling_function()
def has_type_patf(x):
    return RELEVANT if x.d_typ_patf > 0 else ABSTAIN


@labeling_function()
def has_type_sosy(x):
    return RELEVANT if x.d_typ_sosy > 0 else ABSTAIN


@labeling_function()
def has_type_dora(x):
    return RELEVANT if x.d_typ_dora > 0 else ABSTAIN


@labeling_function()
def has_type_fndg(x):
    return RELEVANT if x.d_typ_fndg > 0 else ABSTAIN


@labeling_function()
def has_type_menp(x):
    return RELEVANT if x.d_typ_menp > 0 else ABSTAIN


@labeling_function()
def has_type_chem(x):
    return RELEVANT if x.d_typ_chem > 0 else ABSTAIN


@labeling_function()
def has_type_orch(x):
    return RELEVANT if x.d_typ_orch > 0 else ABSTAIN


@labeling_function()
def has_type_horm(x):
    return RELEVANT if x.d_typ_horm > 0 else ABSTAIN


@labeling_function()
def has_type_phsu(x):
    return RELEVANT if x.d_typ_phsu > 0 else ABSTAIN


@labeling_function()
def has_type_medd(x):
    return RELEVANT if x.d_typ_medd > 0 else ABSTAIN


@labeling_function()
def has_type_bhvr(x):
    return RELEVANT if x.d_typ_bhvr > 0 else ABSTAIN


@labeling_function()
def has_type_diap(x):
    return RELEVANT if x.d_typ_diap > 0 else ABSTAIN


@labeling_function()
def has_type_bacs(x):
    return RELEVANT if x.d_typ_bacs > 0 else ABSTAIN


@labeling_function()
def has_type_enzy(x):
    return RELEVANT if x.d_typ_enzy > 0 else ABSTAIN


@labeling_function()
def has_type_inpo(x):
    return RELEVANT if x.d_typ_inpo > 0 else ABSTAIN


@labeling_function()
def has_type_elii(x):
    return RELEVANT if x.d_typ_elii > 0 else ABSTAIN

@labeling_function()
def has_type_diap_medd_or_bhvr(x):
    return RELEVANT if x.d_typ_diap > 0 or x.d_typ_medd > 0 or x.d_typ_bhvr > 0 else ABSTAIN

@labeling_function()
def number_relations_total(x):
    return RELEVANT if len(x.relationships_list) > 3 else NOT_RELEVANT


@labeling_function()
def number_relations_distinct(x):
    entities = set(x.relationships_list)
    # if 'isa' in entities:
        # entities.remove('isa')
    return RELEVANT if len(entities) > 0 else NOT_RELEVANT


@labeling_function()
def has_treatment(x):
    return RELEVANT if 'treats' in x.relationships_list or 'indicates' in x.relationships_list else ABSTAIN
'''
['query_category', 'query_thread', 'query_text', 'query_annotations', 'typ_dsyn', 'typ_patf', 'typ_sosy', 'typ_dora', 
'typ_fndg', 'typ_menp', 'typ_chem', 'typ_orch', 'typ_horm', 'typ_phsu', 'typ_medd', 'typ_bhvr', 'typ_diap', 'typ_bacs',
 'typ_enzy', 'typ_inpo', 'typ_elii', 'document_category', 'document_thread', 'document_text', 'document_is_doctor_reply',
  'document_number_votes_h', 'document_number_votes_s', 'document_number_votes_t', 'document_user_status', 
  'document_annotations', 
  
  
  'd_typ_dsyn', # disease or syndrome
  'd_typ_patf', # pathological function
  'd_typ_sosy', # sign or syndrome
  'd_typ_dora', # daily or recreational activity 
  'd_typ_fndg', # finding
  *'d_typ_menp', # mental process
  'd_typ_chem', # chemical
  'd_typ_orch', # organic chemical
  'd_typ_horm', # hormone
  'd_typ_phsu', # pharmacological substance
  'd_typ_medd', # medical device
  *'d_typ_bhvr', # behaviour
  *'d_typ_diap', # diagnostic procedure
  'd_typ_bacs', # biologically active substance
  'd_typ_enzy', # enzyme
  'd_typ_inpo', # injury or poisoning
  'd_typ_elii', # element, ion or isotope
  
  
  'bm25_relevant', 'bm25_score']
'''


def classify_my_probs(x):
    return 1 if x < PROBABILITY_CUTOFF else 0


# print(df.document_user_status.unique())
df_train, df_valid = train_test_split(df, test_size=0.1, random_state=8886, stratify=df.bm25_relevant)
Y_valid = df_valid.bm25_relevant.values
Y_train = df_train.bm25_relevant.values

# lfs = [is_long, has_votes, is_doctor_reply, is_same_thread, has_entities, enity_overlap]
lfs = [has_votes, is_doctor_reply, is_same_thread, has_entities, number_relations_distinct, enity_overlap]

# lfs = [is_long, has_votes, is_doctor_reply, is_same_thread, enity_overlap, has_type_dsyn, has_type_patf, has_type_sosy,
#        has_type_dora, has_type_fndg, has_type_menp, has_type_chem, has_type_orch, has_type_horm, has_type_phsu,
#        has_type_medd, has_type_bhvr, has_type_diap, has_type_bacs, has_type_enzy, has_type_inpo, has_type_elii]
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
# label_model.fit(L_train=L_train, n_epochs=20, lr=0.0001, log_freq=10, seed=81794)

valid_probabilities = label_model.predict_proba(L=L_valid)
# validation_labels = np.fromfunction(classify_my_probs, valid_probabilities)
validation_labels = np.fromiter((classify_my_probs(probab[1]) for probab in valid_probabilities), int)

print(average_precision_score(Y_valid, np.transpose(valid_probabilities)[0]))
# joined_vals = np.column_stack((validation_labels, Y_valid))
joined_vals = np.array([validation_labels, Y_valid])
joined_vals = np.vstack((joined_vals, np.zeros_like(joined_vals[0])))

for index, value in enumerate(joined_vals[2]):
    joined_vals[2, index] = index % 20

joined_vals2 = joined_vals[:, joined_vals[2] < METRIC_AT]


print("Accuracy: " + str(accuracy_score(joined_vals[0], joined_vals[1])))
print("Precision: " + str(precision_score(joined_vals2[0], joined_vals2[1])))


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
