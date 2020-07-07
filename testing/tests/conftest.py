"""
Common test configuration logic
"""
import os
import logging
import subprocess
import sys
import time
import timeit

import requests

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s]: {} %(levelname)s %(name)s %(message)s'.format(os.getpid()),
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])

# switch work dir to project root in case tests are run directly
os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/../..')
import pytest


@pytest.fixture()
def check_dataset():
    assert os.path.exists('../data/renditions/')

    def teardown():
        pass


@pytest.fixture()
def testapp(request):
    from api import APP
    app = APP

    client = app.test_client()

    def teardown():
        pass

    request.addfinalizer(teardown)
    return client


@pytest.fixture()
def realapp(request):
    api = subprocess.Popen([sys.executable, '-c', 'import api; api.start_dev_server()'])
    requests.get('http://localhost:5000/status')
    time.sleep(3)
    def teardown():
        api.terminate()
    request.addfinalizer(teardown)
    return None
