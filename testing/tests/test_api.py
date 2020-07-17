import pytest
import requests
import config
import flask.json as json

config.VERIFICATION_MODEL_URI = 'http://storage.googleapis.com/verification-models/verification-metamodel-fps2.tar.xz'


class TestAPI:

    @pytest.mark.usefixtures("testapp")
    def test_verification_api(self, testapp):
        ver_res = testapp.post('/verify', json=dict(source='testing/tests/data/master_4s_1080.mp4', orchestratorID='1',
                                                    renditions=[{'uri': 'testing/tests/data/rend_4s_720_bw.mp4'}]))
        assert 200 == ver_res.status_code
        assert ver_res.json['results'][0]['tamper'] > 0.5

    @pytest.mark.usefixtures("testapp")
    def test_fps_parameters(self, testapp):
        # test verification with 0 fps
        ver_res = testapp.post('/verify', json=dict(source='testing/tests/data/master_4s_1080.mp4', orchestratorID='1',
                                                    renditions=[{'uri': 'testing/tests/data/master_4s_1080.mp4',
                                                                 'frame_rate': 0}]))
        assert 200 == ver_res.status_code
        assert ver_res.json['results'][0]['tamper'] == 0
        # test verification with fractional fps - should pass, master is 60.0 fps
        ver_res = testapp.post('/verify', json=dict(source='testing/tests/data/master_4s_1080.mp4', orchestratorID='1',
                                                    renditions=[{'uri': 'testing/tests/data/rend_4s_720_bw.mp4',
                                                                 'frame_rate': 60000/1001}]))
        assert 200 == ver_res.status_code
        assert ver_res.json['results'][0]['tamper'] == 1
        assert ver_res.json['results'][0]['frame_rate']

        # test verification with fractional fps - should fail
        ver_res = testapp.post('/verify', json=dict(source='testing/tests/data/master_4s_1080.mp4', orchestratorID='1',
                                                    renditions=[{'uri': 'testing/tests/data/rend_4s_720_bw.mp4',
                                                                 'frame_rate': 59}]))
        assert 200 == ver_res.status_code
        assert ver_res.json['results'][0]['tamper'] == 1
        assert not ver_res.json['results'][0]['frame_rate']

    @pytest.mark.usefixtures("realapp")
    def test_verification_api_file_upload(self, realapp):
        master_file = [('file1', ('master_4s_1080.mp4', open('testing/tests/data/master_4s_1080.mp4', 'rb')))]
        rendition_files = [('file2', ('rend_4s_720_bw.mp4', open('testing/tests/data/rend_4s_720_bw.mp4', 'rb')))]
        data = dict(orchestratorID='1', source='master_4s_1080.mp4', renditions=[{'uri': 'rend_4s_720_bw.mp4'}])
        ver_res = requests.post(f'http://{config.API_HOST}:{config.API_PORT}/verify', data={'json': json.dumps(data)}, files=master_file + rendition_files)
        assert 200 == ver_res.status_code
        assert ver_res.json()['results'][0]['tamper'] > 0.5
