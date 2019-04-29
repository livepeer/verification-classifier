import pandas as pd

def train(metrics_path):
    # Retrieve data from repo
    metrics_df = pd.read_csv(metrics_path)
    metrics_df = metrics_df.drop(['Unnamed: 0'], axis=1)
    
    #metrics_df = metrics_df.drop(metrics_df[metrics_df['temporal_difference-cosine']>0.10].index, axis=0)
    metrics_df['title'] = metrics_df['level_0']

    attack_series = []
    for _, row in metrics_df.iterrows():
        attack_series.append(row['level_1'].split('/')[-2])
        
    metrics_df['attack'] = attack_series
    #metrics_df = metrics_df.drop(metrics_df[metrics_df['temporal_canny-euclidean'] > 1000].index, axis=0)

    # Group values for each title
    metrics = ['temporal_histogram_distance-series','temporal_difference-euclidean', 'vmaf', 'temporal_histogram_distance-euclidean', 'temporal_cross_correlation-euclidean']
    grouped_df = metrics_df.groupby(['level_0'] + metrics + ['attack'], as_index=False).count()
    grouped_df = grouped_df.sort_values(by=['attack'])
    print(grouped_df['temporal_histogram_distance-series'])

train('data-analysis/output/metrics.csv')