import os


# environment variable ENV selects actual config class (e.g. ENV=Dev), explicit environment variables are further override any setting

class BaseConfig():
    VERIFICATION_MODEL_URI = ''
    VERIFICATION_MAX_SAMPLES = 10
    API_HOST = 'localhost'
    API_PORT = 5000
    TEMP_PATH = '/tmp'

class DevConfig(BaseConfig):
    pass

class ProdConfig(BaseConfig):
    pass

class TestConfig(BaseConfig):
    pass
