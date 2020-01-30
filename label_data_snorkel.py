"""
 * Created by filip on 07/11/2019

 Label query-document pairs as relevant or non-relevant, using previously trained Snorkel model.

 IMPORTANT: Set the RELEVANT variable True when evaluating relevant pairs or False for non-relevant.

 Input:
    * Pairs of documents and queries with all necessary details.
    * Trained Snorkel model (trained_model_ehf.lbm)

 Output:
    * qrels - relevance judgements
    * used queries - list of query ID in the produced qrel file.
"""

import os
import pandas as pd
import json
import matplotlib.pyplot as plt


from snorkel.labeling import PandasLFApplier, LabelModel
import labelling_functions as lf

print("Hello there docker!")

RELEVANT = False

if RELEVANT:
    # INPUT
    pairs_path = "/container/filip/json/ehealthforum/trac/snorkel_pairs_200k_ehf.txt"

    # OUTPUT
    qrels_path = "/container/filip/json/ehealthforum/trac/qrels_ehf_rel.txt"
    used_queries_path = "/container/filip/json/ehealthforum/trac/used_queries_relevant.json"

else:
    # INPUT
    pairs_path = "/container/filip/json/ehealthforum/trac/non-relevant_pairs_200k_ehf.txt"

    # OUTPUT
    qrels_path = "/container/filip/json/ehealthforum/trac/qrels_ehf_non.txt"
    used_queries_path = "/container/filip/json/ehealthforum/trac/used_queries_non-relevant.json"


open(qrels_path, 'w+', encoding='utf8')
open(used_queries_path, 'w+', encoding='utf8')

used_queries = {}
i = 0

for chunk in pd.read_csv(pairs_path, sep='\t', header=0,
                         error_bad_lines=False, encoding="ISO-8859–1", chunksize=100000):

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

    print(df.head())

    df['query_annotations'] = df['query_annotations'].astype('str')
    mask = (df['query_annotations'].str.len() > 12)
    df = df.loc[mask]

    Y_data = df.bm25_relevant.values
    print(df.shape)

    lfs = [lf.has_type_diap_medd_or_bhvr, lf.is_doctor_reply, lf.has_votes, lf.enity_overlap_jacc, lf.same_author,
           lf.number_relations_total, lf.entity_types]

    applier = PandasLFApplier(lfs)
    L_data = applier.apply(df=df)

    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.load("trained_model_ehf.lbm")

    valid_probabilities = label_model.predict_proba(L=L_data)

    if 'predicted_prob' in df:
        del df['predicted_prob']
    df['predicted_prob'] = valid_probabilities[:, 1]

    PROBABILITY_CUTOFF = 0.5
    df['predicted_label'] = df['predicted_prob'] >= PROBABILITY_CUTOFF
    df_out = df[df['predicted_label'] == int(RELEVANT)][['query_id', 'document_id']]

    with open(qrels_path, 'a+', encoding='utf8') as output_file:
        for index, row in df_out.iterrows():
            output_file.write(str(row['query_id']) + '\t0\t' + str(row['document_id']) + '\t' + str(int(RELEVANT)) + '\n')

    queries_counts = used_queries_batch = df_out['query_id'].value_counts()
    # plt.hist(queries_counts)
    # plt.savefig('/container/filip/json/ehealthforum/plots/counts_hist_' + str(i) + '.png')

    used_queries.update(queries_counts.to_dict())


# Write out Qrels to file.
with open(used_queries_path, 'a', encoding='utf8') as queries_file:
    json.dump(used_queries, queries_file)
    print('Wrote used queries to: ' + str(used_queries_path))

print('\nYay!')
