"""
Module to test the performance and accuracy of QoE calculations
"""
import sys
from functools import reduce
import time

import streamlit as st
import ffmpeg_quality_metrics
import pandas as pd

sys.path.insert(0, '../scripts/asset_processor/')

from video_asset_processor import VideoAssetProcessor

@st.cache
def load_data(original_asset, renditions_list, metrics_list):
    """
    Function to retrieve data from a given file or URL
    in a Pandas DataFrame.
    nrows limits the amount of data displayed for optimization
    """

    asset_processor = VideoAssetProcessor(original_asset,
                                          renditions_list,
                                          metrics_list,
                                          do_profiling=False
                                          )
    data_df, pixels_df, dimensions_df = asset_processor.process()
    return data_df


def show_data():
    original_asset = {'path':'../stream/sources/vimeo/vimeo_99906662.mp4'}
    renditions_list = [{'path':'../stream/1080_14/1080_14_vimeo_99906662.mp4'}
    ]
    metrics_list = ['temporal_ssim',
                    'temporal_psnr']
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
    # Load 10,000 rows of data into the dataframe.
    data_df = load_data(original_asset, renditions_list, metrics_list)
    # Notify the reader that the data was successfully loaded.
    data_load_state.text('Loading data...done!')
    st.subheader('SCIKIT-LEARN')
    st.write(data_df)
    
    start_time = time.time()
    compared_rendition = renditions_list[0]['path']
    json_ssim = ffmpeg_quality_metrics.calc_ssim_psnr(original_asset['path'], original_asset['path'])
    all_dfs = []

    if "psnr" in json_ssim and "ssim" in json_ssim:
        all_dfs.append(pd.DataFrame(json_ssim["psnr"]))
        all_dfs.append(pd.DataFrame(json_ssim["ssim"]))

    if not all_dfs:
        print("No data calculated!")
        sys.exit(1)

    qoe_df = reduce(lambda x, y: pd.merge(x, y, on='n'), all_dfs)

    cols = qoe_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index("n")))
    qoe_df = qoe_df.reindex(columns=cols)
    elapsed_time = time.time()-start_time

    st.subheader('FFMPEG {}'.format(compared_rendition))
    st.write(qoe_df)

    st.subheader('FFMPEG-SUMMARY')
    st.write(qoe_df.describe())
    print('Elapsed time:', elapsed_time)

if __name__ == '__main__':
    show_data()
