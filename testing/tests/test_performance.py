import random
import cv2
import numpy as np
import pandas as pd
import subprocess
from verifier import verifier
import timeit

pd.options.display.width = 0
pd.set_option('display.max_columns', None)


class TestPerformance:

    def get_verification_and_transcoding_speed(self, source_file, rendition_file, n_samples, n_tests, codec):
        ver_results = []
        transcode_results = []

        for i in range(n_tests):
            tc_start = timeit.default_timer()
            args = ['ffmpeg', '-y', '-threads', '1', '-i', source_file,
                    '-c:v', codec, '-vf', 'scale=-2:720',
                    '-b:v', '2000' + 'K', '-c:a', 'copy', '/tmp/out.mp4'
                    ]
            p = subprocess.Popen(args)
            out, err = p.communicate()
            assert not err
            transcode_results.append(timeit.default_timer() - tc_start)

        verifier.retrieve_models('http://storage.googleapis.com/verification-models/verification-metamodel-fps2.tar.xz')
        for i in range(n_tests):
            ver_start = timeit.default_timer()
            res = verifier.verify(source_file, [{"uri": rendition_file}], False, n_samples, '/tmp/model', False, False)
            ver_results.append(timeit.default_timer() - ver_start)

        ver_time = np.min(ver_results)
        transcode_time = np.min(transcode_results)

        return {'best_verification_time': ver_time, 'best_transcoding_time': transcode_time, 'verification_sd': np.std(ver_results), 'transcoding_sd': np.std(transcode_results)}

    def test_verification_speed(self):
        """
        Sanity test to ensure that transcoding speed is significantly lower than verification
        @return:
        """
        np.random.seed(123)
        random.seed(123)

        n_samples = 30
        n_tests = 3
        codec = 'libx264'

        res_2s = self.get_verification_and_transcoding_speed('testing/tests/data/master_2s.mp4', 'testing/tests/data/rend_2s.mp4', n_samples, n_tests, codec)
        res_4s = self.get_verification_and_transcoding_speed('testing/tests/data/master_4s.mp4', 'testing/tests/data/rend_4s.mp4', n_samples, n_tests, codec)

        print(f'Verification vs transcoding for 2s video (1080 to 720):')
        print(res_2s)

        print(f'Verification vs transcoding for 4s video (1080 to 720):')
        print(res_4s)

        assert res_2s['best_verification_time'] < res_2s['best_transcoding_time']
        assert res_4s['best_verification_time'] < res_4s['best_transcoding_time']



