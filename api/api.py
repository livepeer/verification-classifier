# Minimal app for serving Livepeer verification
from verifier import verify, retrieve_model
from flask import Flask, request, jsonify

import logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

@app.route('/verify', methods=['POST'])
def post_route():
    if request.method == 'POST':

        data = request.get_json()
        
        source_uri = data['source']
        model_uri = data['model']

        model_file, model_name = retrieve_model(model_uri)
        
        do_profiling = False
        max_samples = 10

        predictions = verify(source_uri, data['renditions'], do_profiling, max_samples, model_file, model_name)
        results = []
        i = 0
        for rendition in data['renditions']:
            results.append({rendition['uri'] : predictions[i]})
            i += 1
        return jsonify(results)
  

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')