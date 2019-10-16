"""
Module for collecting metrics values from GCE datastore
generated with the cloud functions located in feature_engineering
USAGE:

$python3 collect_from_datastore.py

This will create a .csv file in the current folder containing
the values of all the metrics available in the database for later
use in the jupyter notebooks of this section
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


# [START datastore_build_service]
from google.cloud import datastore

# Function to create datastore client
def create_client(project_id):
    return datastore.Client(project_id)
# [END datastore_build_service]

# Function to gather all properties for a kind in a pandas dataframe
def get_jobs_df(kind, namespace):
    query = client.query(kind=kind, namespace=namespace)
    query_iter = query.fetch()
    i = 0
    number_of_pages = 300
    jobs_df = pd.DataFrame()
    for page in tqdm(query_iter.pages):
        i += 1
        tasks = list(page)
        page_df = pd.DataFrame(data=tasks)
        print(i * number_of_pages, ' videos retrieved so far')
        jobs_df = pd.concat([jobs_df, page_df], axis=0,sort=True)
    print('Data retrieval completed {} videos retrieved, {} features extracted'.format(jobs_df.shape[0],jobs_df.shape[1]))
    return jobs_df


def initialize():
    global client

    print('Initializing...')
    pd.set_option('display.max_colwidth', -1)

    namespace = 'livepeer-verifier-QoE'
    client = create_client('epiclabs')
    query = client.query(kind='__kind__',namespace=namespace)
    query.keys_only()
    jobs_dict = {}
    inputs_df = pd.DataFrame()

    print('Getting inputs...')
    input_kinds = [entity.key.name for entity in query.fetch() if 'features_input_60_540' in entity.key.name]
    
    print('Retrieving data from Datastore...')
    for kind in input_kinds:
        
        kind_df = get_jobs_df(kind, namespace)
        kind_df['kind'] = kind
        inputs_df = pd.concat([inputs_df, kind_df],axis=0,sort=True, ignore_index=True)

        jobs_dict[kind] = inputs_df['title'][inputs_df['kind']==kind]
    inputs_df.to_csv('data-qoe-large.csv')
initialize()
