"""
 * Created by filip on 07/11/2019

 Train Snorkel classifier. Uses supplied pairs of queries documents, roughly balanced to contain the same amount
 of relevant (according to BM25) and nonrelevant (according to BM25) pais. Saves the trained model into a file.
"""

import os
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, average_precision_score
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis, LabelModel
import labelling_functions as lf


df = pd.read_csv("/container/filip/json/ehealthforum/trac/training_data_snorkel_ehf_100k_titles.txt", sep='\t', header=0,
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
# df = df.iloc[:300, ]
print(df.head())

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


df_train = df

Y_train = df_train.bm25_relevant.values

lfs = [lf.has_type_diap_medd_or_bhvr, lf.is_doctor_reply, lf.has_votes, lf.enity_overlap_jacc, lf.same_author,
       lf.number_relations_total, lf.entity_types]

applier = PandasLFApplier(lfs)

L_train = applier.apply(df=df_train)

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=20000, lr=0.0001, log_freq=10, seed=2345)
label_model.save("trained_model_ehf.lbm")

print("Finished,")
