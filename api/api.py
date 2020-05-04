"""
Minimal app for serving Livepeer verification
"""

import logging
# create formatter to add it to the logging handlers
FORMATTER = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')

import bjoern
from flask import Flask, request, jsonify

from verifier import verify, retrieve_models

APP = Flask(__name__)

def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""
    if log_file == '':
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(FORMATTER)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# Setup console logger
CONSOLE_LOGGER = setup_logger('console_logger', '')
CONSOLE_LOGGER = logging.getLogger('console_logger')
# Setup operations logger
OPERATIONS_LOGGER = setup_logger('operations_logger', 'logs/ops.log')
OPERATIONS_LOGGER = logging.getLogger('operations_logger')
# Setup operations logger
VERIFICATIONS_LOGGER = setup_logger('verifications_logger', 'logs/verifications.log')
VERIFICATIONS_LOGGER = logging.getLogger('verifications_logger')


@APP.route('/verify', methods=['POST'])
def post_route():
    """
    Verification endpoint.

    This function just responds to the api call in localhost:5000/verify
    Input parameters:

    "orchestrator_id": The ID of the orchestrator responsible of transcoding
    "source": a valid URI to a video source
    "renditions": a list of renditions with the following structure:
    {
         "uri":A valid URI toi the transcoded video,
         "resolution":{
            "height": vertical dimension, in pixels
            "width": horizontal dimension, in pixels
         },
         "frame_rate": A value of the expected frames per seconds
         "pixels": The number of expected total pixels
                   (height x width x number of frames)
    },
   "model": The URL to the location of the trained model for verification

    Returns:

    {"model": The url of the inference model used,
     "orchestrator_id": The ID of the orchestrator responsible of transcoding,
     "source": The URI of the video source,
     "results": A list with the verification results, with the following:
        {
                "audio_available": A boolean indicating whether the audio track could be extracted
                                    from the source,
                "frame_rate": The ratio between the expected frame rate and the one extracted
                             with OpenCv's backend (GStreamer by default),
                "ocsvm_dist": A float that represents the distance to the decision function of the
                            One Class Support Vector Machine used for inference. Negative values
                            indicate a tampering has been detected by this model (see tamper_ul),                           
                "pixels": The number of expected total pixels (height x width x number of frames)
                "pixels_pre_verification": The ratio between the expected number of total pixels and
                                           the one extracted with OpenCv's backend
                                           (GStreamer by default)
                "pixels_post_verification": The ratio between the expected number of
                                            total pixels and the one computed during the decoding
                "resolution":
                {
                    "height": The expected total vertical pixels
                    "height_pre_verification": The ratio between the expected
                                                height and the one extracted
                                                with OpenCv's backend (GStreamer by default)
                    "height_post_verification": The ratio between the expected height and the one
                                                computed during the decoding
                    "width": The expected total horizontal pixels
                    "width_pre_verification": The ratio between the expected
                                                width and the one extracted
                                                with OpenCv's backend (GStreamer by default)
                    "width_post_verification":The ratio between the expected height and the one
                                                computed during the decoding
                },
                "ssim_pred": A float representing an estimated measure of the transcoding's SSIM,
                "tamper_meta": A boolean indicating whether the metamodel detected tampering,
                "tamper_sl": A boolean indicating whether the Supervised Learning model detected
                            tampering,
                "tamper_ul": A boolean indicating whether the Unsupervised Learning model detected
                            tampering, 
                "uri": The URI of the rendition
     }
    """
   
    if request.method == 'POST':


        data = request.get_json()

        verification = {}

        verification['orchestrator_id'] = data['orchestratorID']
        verification['source'] = data['source']

        model_uri = data['model']
        # TODO: Uncomment this line for testing of the model using Livepeer's Broadcaster node
        # Once the broadcaster allows for model selection, remove the line below
        # model_uri = 'https://storage.googleapis.com/verification-models/verification-metamodel.tar.xz'
        model_dir, model_name_ul, model_name_sl, model_name_qoe = retrieve_models(model_uri)

        # Inform user that model was succesfully retrieved
        OPERATIONS_LOGGER.info('Model successfully downloaded: %s', model_uri)
        CONSOLE_LOGGER.info('Model successfully downloaded: %s', model_uri)

        # Define whether profiling is needed for logging
        do_profiling = False
        # Define the maximum number of frames to sample
        max_samples = 10

        # Execute the verification
        predictions = verify(verification['source'],
                             data['renditions'],
                             do_profiling,
                             max_samples,
                             model_dir,
                             model_name_ul,
                             model_name_sl,
                             model_name_qoe)
        results = []
        i = 0
        for _ in data['renditions']:
            results.append(predictions[i])
            i += 1

        # Append the results to the verification object
        verification['results'] = results
        verification['model'] = model_uri

        VERIFICATIONS_LOGGER.info(verification)
        CONSOLE_LOGGER.info('Verification results: %s', results)

        return jsonify(verification)

if __name__ == '__main__':
    HOST = '0.0.0.0'
    PORT = 5000

    CONSOLE_LOGGER.info('Verifier server listening in port %s', PORT)
    OPERATIONS_LOGGER.info('Verifier server listening in port %s', PORT)

    bjoern.listen(APP, HOST, PORT)
    bjoern.run()
