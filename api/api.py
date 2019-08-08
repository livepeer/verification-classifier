# Minimal app for serving Livepeer verification
from verifier import verify
from flask import Flask, request

import urllib
import os
import uuid
import tarfile

app = Flask(__name__)

@app.route('/verify', methods=['POST'])
def post_route():
    if request.method == 'POST':

        data = request.get_json()
        renditions = []
        source_uri = data['source']
        model_uri = data['model']

        source_file = retrieve_video_file(source_uri)
        model_file, model_name = retrieve_model(model_uri)

        for rendition_uri in data['renditions']:
            video_file = retrieve_video_file(rendition_uri)
            renditions.append(video_file)
        
        do_profiling = False
        max_samples = 10

        predictions = verify(source_file, renditions, do_profiling, max_samples, model_file, model_name)
        results = []
        i = 0
        for rendition_uri in data['renditions']:
            results.append({rendition_uri : predictions[i]})
            i += 1
        return 'Results: {}\n'.format(results)

def retrieve_video_file(uri):
    if 'http' in uri:
        file_name = '/tmp/{}'.format(uuid.uuid4())
        
        print('File download started!')
        video_file, _ = urllib.request.urlretrieve(url, filename=file_name)
        
        print('File downloaded')
    else:
        video_file = uri
    return video_file

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
    app.run(debug=True, host='0.0.0.0')