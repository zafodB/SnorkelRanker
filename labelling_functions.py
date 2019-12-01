'''
 * Created by filip on 29/11/2019
'''

from snorkel.labeling import labeling_function

doc_column_list = [
    'd_typ_dsyn',  # disease or syndrome
    'd_typ_patf',  # pathological function
    'd_typ_sosy',  # sign or syndrome
    'd_typ_dora',  # daily or recreational activity
    'd_typ_fndg',  # finding
    'd_typ_menp',  # mental process
    'd_typ_chem',  # chemical
    'd_typ_orch',  # organic chemical
    'd_typ_horm',  # hormone
    'd_typ_phsu',  # pharmacological substance
    'd_typ_medd',  # medical device
    'd_typ_bhvr',  # behaviour
    'd_typ_diap',  # diagnostic procedure
    'd_typ_bacs',  # biologically active substance
    'd_typ_enzy',  # enzyme
    'd_typ_inpo',  # injury or poisoning
    'd_typ_elii',  # element, ion or isotope
]

query_column_list = [
    'typ_dsyn',  # disease or syndrome
    'typ_patf',  # pathological function
    'typ_sosy',  # sign or syndrome
    'typ_dora',  # daily or recreational activity
    'typ_fndg',  # finding
    'typ_menp',  # mental process
    'typ_chem',  # chemical
    'typ_orch',  # organic chemical
    'typ_horm',  # hormone
    'typ_phsu',  # pharmacological substance
    'typ_medd',  # medical device
    'typ_bhvr',  # behaviour
    'typ_diap',  # diagnostic procedure
    'typ_bacs',  # biologically active substance
    'typ_enzy',  # enzyme
    'typ_inpo',  # injury or poisoning
    'typ_elii',  # element, ion or isotope
]

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
    return RELEVANT if text_length < 1400 else ABSTAIN


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
def enity_overlap_coeff(x):
    document_annotations = set(x.document_annotations)
    query_annotations = set(x.query_annotations)

    smaller = document_annotations if len(document_annotations) < len(query_annotations) else query_annotations

    if len(smaller) == 0:
        return NOT_RELEVANT
    elif len(document_annotations.intersection(query_annotations)) / len(smaller) >= 1:
        return RELEVANT
    else:
        return NOT_RELEVANT


@labeling_function()
def enity_overlap_jacc(x):
    document_annotations = set(x.document_annotations)
    query_annotations = set(x.query_annotations)

    if len(document_annotations.union(query_annotations)) == 0:
        return NOT_RELEVANT
    elif len(document_annotations.intersection(query_annotations)) / len(
        document_annotations.union(query_annotations)) > 0.3:
        return RELEVANT
    else:
        return NOT_RELEVANT


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


@labeling_function()
def entity_types(x):
    various_types = 0

    for column in doc_column_list:
        if x[column] > 1:
            various_types += 1

    return RELEVANT if various_types > 0 else NOT_RELEVANT


@labeling_function()
def entity_type_overlap(x):
    doc_entities = set()
    query_entities = set()

    for column in doc_column_list:
        if x[column] > 0:
            # print(column[2:])
            doc_entities.add(column[2:])

    for column in query_column_list:
        if x[column] > 0:
            query_entities.add(column)

    # print(len(doc_entities.intersection(query_entities)))
    return RELEVANT if len(doc_entities.intersection(query_entities)) > 1 else NOT_RELEVANT


@labeling_function()
def entity_type_overlap_fraction(x):
    doc_entities = set()
    query_entities = set()

    for column in doc_column_list:
        if x[column] > 0:
            # print(column[2:])
            doc_entities.add(column[2:])

    for column in query_column_list:
        if x[column] > 0:
            query_entities.add(column)

    if len(doc_entities) + len(query_entities) == 0:
        return NOT_RELEVANT
    elif len(doc_entities.intersection(query_entities)) / (len(doc_entities) + len(query_entities) ) > 0.3:
        return RELEVANT
    else:
        return NOT_RELEVANT


@labeling_function()
def entity_type_overlap_jacc(x):
    doc_entities = set()
    query_entities = set()

    for column in doc_column_list:
        if x[column] > 0:
            # print(column[2:])
            doc_entities.add(column[2:])

    for column in query_column_list:
        if x[column] > 0:
            query_entities.add(column)

    if len(doc_entities.union(query_entities)) == 0:
        return NOT_RELEVANT
    elif len(doc_entities.intersection(query_entities)) / (len(doc_entities.union(query_entities))) > 0.6:
        return RELEVANT
    else:
        return NOT_RELEVANT
