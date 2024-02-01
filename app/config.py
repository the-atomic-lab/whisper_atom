import os
from os import environ
from distutils.util import strtobool



class EnvVar(object):
    APP_VERSION = "1.0.0"
    APP_NAME = "ASR"
    API_PREFIX = "/audio"
    IS_DEBUG = bool(os.environ.get('IS_DEBUG', 0))
    
    MAX_NUM_INDEX = int(os.environ.get('MAX_NUM_INDEX', 1000))
    
    R2DB_URL = os.environ.get('R2BASE_URL', 'http://192.1.2.230:8000')
    
    N_WORKERS_SERVING = int(os.environ.get('N_WORKERS_SERVING', 1))
    
    #   可调参数
    STT_INFER_BATCH = int(os.environ.get('STT_INFER_BATCH', 16))
    COMPUTE_TYPE = os.environ.get('COMPUTE_TYPE', "float16")