import click
import urllib
import sys

sys.path.insert(0, 'scripts/asset_processor')
# Minimal app for serving Livepeer verification
from verifier import retrieve_model, verify, pre_verify

@click.command()
@click.argument('source')
@click.argument('renditions', multiple=True)
@click.argument('model_uri')
@click.argument('pre_verification_parameters')
@click.option('--do_profiling', default=0)
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
    # -model_uri:                   Path to location in disk where pre-traind model is located
    # ************************************************************************

    model_dir, model_file = retrieve_model(model_uri)

    predictions = verify(source, renditions, do_profiling, max_samples, model_dir, model_file)
    results = []
    i = 0
    for rendition_uri in renditions:
        results.append({rendition_uri : predictions[i]})
        i += 1
    return 'Results: {}\n'.format(results)


if __name__ == '__main__':
    cli()
