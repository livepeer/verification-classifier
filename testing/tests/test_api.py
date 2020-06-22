import pytest
import config
config.VERIFICATION_MODEL_URI = 'http://storage.googleapis.com/verification-models/verification-metamodel-fps2.tar.xz'

class TestAPI:

    @pytest.mark.usefixtures("testapp")
    def test_verification_api(self, testapp):

        ver_res = testapp.post('/verify', json=dict(source='testing/tests/data/master_4s_1080.mp4', orchestratorID='1',
                                                    renditions=[{'uri': 'testing/tests/data/rend_4s_720_bw.mp4'}]))
        assert 200 == ver_res.status_code
        assert ver_res.json['results'][0]['tamper'] > 0.5
