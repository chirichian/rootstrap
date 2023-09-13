"""
Knowledge Base
--------------
This file containst all knowledge base of the project. This includes:
- Constants
- Static tables (E.g: to store Table names or codes)
"""
import numpy as np

from utils import data_path


class ParamsValues(object):
    features_static_values = {
        'valid_modeling_years': [
            2010,
            2011,
            2012,
            2013, 
            2014, 
            2015,
            2016, 
            2017,
            2018, 
            2019, 
        ],
    }

    training_params = {
        'input_train': data_path('processed', 'train_data.csv'),
        'input_val': data_path('processed', 'test_data.csv'),
        'output': {
            'folders': [
                data_path('modeling', 'output'),
            ],
            'arima_model': data_path('modeling', 'model.joblib'),
            'holt_model': data_path('modeling', 'model.joblib'),
        },
        'model': {
            'arima_params': {
                'start_p':0, 
                'd':1,
                'start_q':0, 
                'max_p':5, 
                'max_d':5,
                'max_q':5,
                'start_P':0,
                'start_Q':0,
                'D':1,
                'max_D':5,
                'max_P':5,
                'max_Q':5,
                'seasonal':True,
                'm':12,
                'random_state':5
            },
            'holt_params':{
                'seasonal_periods':12 ,
                'trend':'add', 
                'seasonal':'add'
            }
        }
    }

    predict_params = {
            'best_model': data_path('modeling', 'model.joblib'),
        }