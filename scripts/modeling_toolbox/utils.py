import os
import pandas as pd

def load_data(data_uri, nrows):
    """
    Function to retrieve data from a given file or URL
    in a Pandas DataFrame suitable for model training.
    nrows limits the amount of data
    """
    data_df = pd.read_csv(data_uri, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data_df.rename(lowercase, axis='columns', inplace=True)
    data_df.rename(columns={'attack':'rendition',
                            'title':'source',
                            'dimension':'dimension_x'},
                   inplace=True)
    data_df['rendition'] = data_df['rendition'].apply(lambda x: set_rendition_name(x))
    data_df['dimension_y'] = data_df['rendition'].apply(lambda x: int(x.replace('p','').split('_')[0]))
    data_df['size_dimension_ratio'] = data_df['size'] / (data_df['dimension_y'] * data_df['dimension_x'])
    data_df['target'] = data_df['rendition'].str.contains('^[0-9]+p(_[0-9][0-9]?-[0-9][0-9]?fps(_gpu)?)?$')
    return data_df

def set_rendition_name(rendition_name):
    """
    Function to extract source name from rendition path
    """
    try:
        return os.path.dirname(rendition_name).replace('/vimeo', '').split('/')[-1]
    except:
        return ''

if __name__=='__main__':
	load_data('../../data/data-large.csv', nrows=100)