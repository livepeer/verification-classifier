import os
import pandas as pd
import tqdm
import numpy as np

def load_data(data_uri, nrows=None):
    """
    Function to retrieve data from a given file or URL
    in a Pandas DataFrame suitable for model training.
    nrows limits the amount of data
    """
    data_df = pd.read_csv(data_uri, nrows=nrows)
    data_df.index = data_df.id
    lowercase = lambda x: str(x).lower()
    data_df.rename(lowercase, axis='columns', inplace=True)
    data_df.rename(columns={'attack':'rendition',
                            'title':'source'},
                   inplace=True)
    data_df['rendition'] = data_df['rendition'].apply(lambda x: set_rendition_name(x))
    data_df['target'] = np.logical_not(data_df['rendition'].str.contains('^[0-9]+p(_[0-9][0-9]?-[0-9][0-9]?fps(_gpu)?)?$')).astype(np.int32)
    data_df['master_id'] = data_df.id.str.extract('/(.+)')
    return data_df

def set_rendition_name(rendition_name):
    """
    Function to extract source name from rendition path
    """
    try:
        return os.path.dirname(rendition_name).replace('/vimeo', '').split('/')[-1]
    except:
        return ''

def update_with_qoe_data(src_dataset, qoe_dataset):
    src = pd.read_csv(src_dataset)
    qoe = pd.read_csv(qoe_dataset)
    qoe['id'] = qoe.attack.str.replace('/tmp/','')
    qoe.index = qoe.id
    src.index = src.id
    cols_qoe = ['temporal_ssim-mean']
    for idx in tqdm.tqdm(src.index.intersection(qoe.index)):
        if pd.isna(src.loc[idx, 'temporal_ssim-mean']) and idx in qoe.index:
            src.loc[idx, 'temporal_ssim-mean'] = qoe.loc[idx, 'temporal_ssim-mean']
    src.to_csv(src_dataset)
    pass


if __name__=='__main__':
    update_with_qoe_data('../../data/data-large.csv', '../../data/data-qoe-large.csv')
    #data = load_data('../../data/data-large.csv', nrows=100)
