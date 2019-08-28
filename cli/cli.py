import click
import urllib
import sys

sys.path.insert(0, 'scripts/asset_processor')
# Minimal app for serving Livepeer verification
from verifier import retrieve_model, verify, pre_verify

@click.command()
@click.argument('source')
@click.argument('model_uri')
@click.option('--renditions', multiple=True)
@click.option('--max_samples', type=int, default=10)
@click.option('--do_profiling', default=0)
def cli(source, renditions, do_profiling, max_samples, model_uri):
    
    model_file, model_name = retrieve_model(model_uri)

    predictions = verify(source, renditions, do_profiling, max_samples, model_file, model_name)
    results = []
    i = 0
    for rendition_uri in renditions:
        results.append({rendition_uri : predictions[i]})
        i += 1
    return 'Results: {}\n'.format(results)

if __name__ == '__main__':
    cli()