import click
import urllib
import sys
import tarfile
import os

sys.path.insert(0, 'scripts/asset_processor')
# Minimal app for serving Livepeer verification
from verifier import verify

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

def retrieve_model(uri):
    model_dir = '/tmp/model'
    model_file = uri.split('/')[-1]
    # Create target Directory if don't exist
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        print("Directory " , model_dir ,  " Created ")
        print('Model download started!')
        filename, _ = urllib.request.urlretrieve(uri, filename='{}/{}'.format(model_dir, model_file))
        print('Model downloaded')
        try:
            with tarfile.open(filename) as tf:
                tf.extractall(model_dir)
                return model_dir, model_file
        except Exception:
            return 'Unable to untar model',''
    else:    
        print("Directory " , model_dir ,  " already exists, skipping download")
        return model_dir, model_file


if __name__ == '__main__':
    cli()