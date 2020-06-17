"""
Common test configuration logic
"""
import os
import logging

logging.basicConfig(level=logging.INFO,
					format='[%(asctime)s]: {} %(levelname)s %(name)s %(message)s'.format(os.getpid()),
					datefmt='%Y-%m-%d %H:%M:%S',
					handlers=[logging.StreamHandler()])

# switch work dir to project root in case tests are run directly
os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/../..')
import pytest


@pytest.fixture()
def check_dataset():
    assert os.path.exists('../data/renditions/') and \
           os.path.exists('../data/yt8m-large-train.csv') and \
           os.path.exists('../data/yt8m-large-test.csv')
    def teardown():
        pass
