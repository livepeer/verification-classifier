import numpy as np
import pandas as pd

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
    jobs_df = pd.DataFrame()
    for page in query_iter.pages:
        tasks = list(page)
        page_df = pd.DataFrame(data=tasks)
        jobs_df = pd.concat([jobs_df, page_df], axis=0,sort=True)
        
    return jobs_df


def initialize():
    global client

    print('Initializing...')
    pd.set_option('display.max_colwidth', -1)

    namespace = 'livepeer-verifier-training'
    client = create_client('epiclabs')
    query = client.query(kind='__kind__',namespace=namespace)
    query.keys_only()
    jobs_dict = {}
    inputs_df = pd.DataFrame()

    print('Getting inputs...')
    input_kinds = [entity.key.name for entity in query.fetch() if 'asset_input' in entity.key.name]
    for kind in input_kinds:
        kind_df = get_jobs_df(kind, namespace)
        kind_df['kind'] = kind
        inputs_df = pd.concat([inputs_df, kind_df],axis=0,sort=True, ignore_index=True)

        jobs_dict[kind] = inputs_df['title'][inputs_df['kind']==kind]
    inputs_df.to_csv('data.csv')
initialize()
