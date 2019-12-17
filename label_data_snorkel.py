'''
 * Created by filip on 07/11/2019
'''

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

print("Hello there docker!")

open("/container/filip/json/ehealthforum/trac/qrels_ehf_0.txt", 'w+', encoding='utf8')

i = 0
for chunk in pd.read_csv("/container/filip/json/ehealthforum/trac/training_data_ehf_mt_200k.txt", sep='\t', header=0,
                         error_bad_lines=True, encoding="ISO-8859–1", chunksize=100000):

    df = chunk
    print('Now processing chunk number ' + str(i))
    i += 1


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

    df_data = df

    # df_valid = df.iloc[3000:, ]
    # df_train = df.iloc[:3000, ]

    Y_data = df_data.bm25_relevant.values
    print(df_data.shape)

    lfs = [lf.has_type_diap_medd_or_bhvr, lf.is_doctor_reply, lf.has_votes, lf.enity_overlap_jacc, lf.same_author,
           lf.number_relations_total, lf.entity_types]

    applier = PandasLFApplier(lfs)

    L_data = applier.apply(df=df_data)

    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.load("/container/filip/json/ehealthforum/trac/trained_model_ehf.lbm")

    valid_probabilities = label_model.predict_proba(L=L_data)

    if 'predicted_prob' in df_data:
        # df_valid.drop(columns=['predicted_prob'], axis=1)
        del df_data['predicted_prob']

    df_data['predicted_prob'] = valid_probabilities[:, 1]

    PROBABILITY_CUTOFF = 0.5
    df_data['predicted_label'] = df_data['predicted_prob'] >= PROBABILITY_CUTOFF

    print(df_data['predicted_label'] == 1)
    df_out = df_data[df_data['predicted_label'] == 1][['query_id', 'document_id']]

    with open("/container/filip/json/ehealthforum/trac/qrels_ehf_0.txt", 'a+', encoding='utf8') as output_file:
        for index, row in df_out.iterrows():
            output_file.write(str(row['query_id']) + '\t0\t' + str(row['document_id']) + '\t1\n')

print('\nsup')
