import click
import sys

sys.path.insert(0, 'scripts/asset_processor')
# Minimal app for serving Livepeer verification
from verifier import retrieve_model, verify, pre_verify


@click.command()
@click.argument('source')
@click.option('--renditions', multiple=True)
@click.argument('model_uri')
@click.option('--pre_verification_parameters')
@click.option('--do_profiling', default=0)
<<<<<<< HEAD
@click.option('--max_samples', type=int, default=10)
def cli(source, renditions, model_uri, do_profiling, max_samples):
    # ************************************************************************
    # Function to aggregate predicted verifications from a pre-trained model,
    # a source video asset and a list of its renditions.
    # Arguments:
    # -source:                      The source video asset from which renditions were created
    # -renditions:                  A list of paths in disk where renditions are located
    # -do_profiling:                Enables / disables profiling tools for debugging purposes
    # -max_samples:                 Number o random sampling frames to be used to make predictions
    # -model_uri:                   Path to location in disk where trained model is located
    # ************************************************************************

    model_dir, model_file = retrieve_model(model_uri)

    predictions = verify(source, renditions, do_profiling, max_samples, model_dir, model_file)
    results = []
=======
def cli(asset, renditions, do_profiling):
    # Download model from remote url
    total_start = time.clock()
    model_url = 'https://github.com/livepeer/verification-classifier/blob/master/machine_learning/output/models/' \
                'model.tar.gz?raw=true'
    model_name = 'XGBoost'
    scaler_type = 'MinMaxScaler'
    learning_type = 'SL'
    start = time.clock()
    download_models(model_url)
    download_time = time.clock() - start
    loaded_model = pickle.load(open('{}.pickle.dat'.format(model_name), 'rb'))
    loaded_scaler = pickle.load(open('{}_{}.pickle.dat'.format(learning_type, scaler_type), 'rb'))

    with open('param_{}.json'.format(model_name)) as json_file:
        params = json.load(json_file)
        features = params['features']

    # Prepare input variables
    original_asset = asset
    renditions_list = list(renditions)
    metrics_list = ['temporal_gaussian']

    # Process and compare original asset against the provided list of renditions
    start = time.clock()
    asset_processor = video_asset_processor(original_asset, renditions_list, metrics_list, 4, do_profiling)
    initialize_time = time.clock() - start

    start = time.clock()
    metrics_df = asset_processor.process()
    process_time = time.clock() - start

    # Cleanup the resulting pandas dataframe and convert it to a numpy array
    # to pass to the prediction model
    for column in metrics_df.columns:
        if 'series' in column:
            metrics_df = metrics_df.drop([column], axis=1)

    features.remove('attack_ID')

    metrics_df = metrics_df[features]
    metrics_df = metrics_df.drop('title', axis=1)
    metrics_df = metrics_df.drop('attack', axis=1)
    X = np.asarray(metrics_df)
    # Scale data:
    X = loaded_scaler.transform(X)

    matrix = pickle.load(open('reduction_{}.pickle.dat'.format(model_name), 'rb'))
    X = matrix.transform(X)

    # Make predictions for given data
    start = time.clock()
    y_pred = loaded_model.predict(X)
    prediction_time = time.clock() - start

    # Display predictions
>>>>>>> Removes unneeded features from cli.py
    i = 0
    for rendition_uri in renditions:
        results.append({rendition_uri: predictions[i]})
        i += 1
    return 'Results: {}\n'.format(results)


if __name__ == '__main__':
    cli()
