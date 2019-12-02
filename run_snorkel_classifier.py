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
import labelling_functions as lf

print("Hello there docker!")

# METRIC_AT = 5
# PROBABILITY_CUTOFF = 0.1

# df = pd.read_csv("/filip/json/ehealthforum/trac/training_data.txt", sep='\t',
#                  names=['mdreply', 'votesh', 'votess', 'votest', 'ment', 'sameent', 'length', 'category', 'thread'])

# import os
# for root, dirs, files in os.walk('/container'):
#     for direc in dirs:
#         print(os.path.join(root, direc))

df = pd.read_csv("/container/filip/json/ehealthforum/trac/training_data_snorkel_10k_titles.txt", sep='\t', header=0,
                 error_bad_lines=True, encoding="ISO-8859–1")


def split_by_char(input_text, character=';'):
    try:
        return input_text.split(sep=character)
    except AttributeError:
        return ""



df['document_annotations'] = df['document_annotations'].apply(split_by_char)
df['query_annotations'] = df['query_annotations'].apply(split_by_char)
df.relationships_list = df.relationships_list.apply(split_by_char, args=(',',))

df = df.sort_values(['query_thread', 'bm25_score'], ascending=[True, False])

# Keep just a bit of data for development for now
# df = df.iloc[:1000, ]
print(df.head())
# print()
# print(df.bm25_score.info())
# print(df.bm25_score.describe())
# print()
# print(df.shape)
# print(df.ndim)

# region Labeling functions
# Define the label mappings for convenience


# endregion


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


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


# df_train, df_valid = train_test_split(df, test_size=0.1, random_state=8886, stratify=df.query_thread)
# df_train, df_valid = train_test_split(df, test_size=0.5, shuffle=False)
# df_train, df_valid = np.array_split(df, 2)
df_valid = df.iloc[70000:, ]
df_train = df.iloc[:70000, ]

Y_valid = df_valid.bm25_relevant.values
Y_train = df_train.bm25_relevant.values

lfs = [lf.has_votes, lf.is_doctor_reply, lf.is_same_thread, lf.has_entities, lf.enity_overlap, lf.enity_overlap_jacc, lf.entity_type_overlap_jacc]
# lfs = [is_same_thread, enity_overlap, entity_types, entity_type_overlap]

# lfs = [is_long, has_votes, is_doctor_reply, is_same_thread, enity_overlap, has_type_dsyn, has_type_patf, has_type_sosy,
#        has_type_dora, has_type_fndg, has_type_menp, has_type_chem, has_type_orch, has_type_horm, has_type_phsu,
#        has_type_medd, has_type_bhvr, has_type_diap, has_type_bacs, has_type_enzy, has_type_inpo, has_type_elii]
# lfs = [has_votes, is_doctor_reply, is_same_thread, enity_overlap]
# lfs = [is_same_thread, enity_overlap, is_doctor_reply]

applier = PandasLFApplier(lfs=lfs)

L_train = applier.apply(df=df_train)
L_valid = applier.apply(df=df_valid)

analysis = LFAnalysis(L=L_train, lfs=lfs).lf_summary(Y=Y_train)
print(analysis)
print(analysis['Conflicts'])
print(analysis['Overlaps'])

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=20000, lr=0.0001, log_freq=10, seed=2345)
# label_model.fit(L_train=L_train, n_epochs=20, lr=0.0001, log_freq=10, seed=81794)

print("Model weights: " + str(label_model.get_weights()))

valid_probabilities = label_model.predict_proba(L=L_valid)
df_valid.insert(50, 'predicted_prob', valid_probabilities[:, 1])

df_valid.to_csv("/filip/json/ehealthforum/trac/validation_df2.txt", sep="\t", header=True)
# df_valid = pd.read_csv("/filip/json/ehealthforum/trac/validation_df.txt", sep="\t")

#
METRIC_AT = 5


# PROBABILITY_CUTOFF = 0.1

#
#
def compute_precision_at_k(l, k):
    l = l[:k]
    return sum(l) / k


# for cutoff in range(1, 10):

# PROBABILITY_CUTOFF = (cutoff / 10)
PROBABILITY_CUTOFF = 0.5

df_valid['predicted_label'] = df_valid['predicted_prob'] >= PROBABILITY_CUTOFF

mapk_true = (df_valid.sort_values(['query_thread', 'bm25_score'], ascending=[True, False])
             .groupby('query_thread')
             .head(10)
             .groupby('query_thread')['document_text']
             .apply(list)
             .tolist())

mapk_predicted = (df_valid.sort_values(['query_thread', 'predicted_prob'], ascending=[True, False])
                  .groupby('query_thread')
                  .head(10)
                  .groupby('query_thread')['document_text']
                  .apply(list)
                  .tolist())

print("\nCutoff value (labelled True if above): " + str(PROBABILITY_CUTOFF) + '\nMetrics@: ' + str(METRIC_AT) + '\n')

print("Number of True relevant: " + str(df_valid[df_valid.bm25_relevant == 1].count()['bm25_relevant']))
print("Number of Predicted relevant: " + str(
    df_valid[df_valid.predicted_label == 1].count()['predicted_label']) + '\n')

df_tru = df_valid.groupby(['query_thread']).head(10)['bm25_relevant']

df_pred = df_valid.groupby(['query_thread']).head(10)['predicted_label']

overall_precision = []

for query, group in df_valid.groupby(['query_thread']):
    precision = compute_precision_at_k(group['predicted_label'].head(10).tolist(), 10)
    overall_precision.append(precision)

print('Overall precision: ' + str(sum(overall_precision) / len(overall_precision)))
print("Accuracy: " + str(accuracy_score(df_tru, df_pred)))
# print("Precision: " + str(compute_precision_at_k(df_pred, 10)))
print("MAP@" + str(METRIC_AT) + ': ' + str(mapk(mapk_true, mapk_predicted, METRIC_AT)))

#
label_model_acc = label_model.score(L=L_valid, Y=Y_valid)["accuracy"]
print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")
#
#
print('\nsup')
#
