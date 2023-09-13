"""Script to run churn model training"""

import ast
from os.path import exists
from os import makedirs
import os
from matplotlib import pyplot as plt
from utils import data_path

import click
import logging
import pandas as pd
import json

from models import model
from kb import ParamsValues

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command(name='train')
@click.option('--arima_params',
              default='{}',
              help=('arima params. Must be a dict in a string format. ' +
                    'Example: --arima_params "{\'num_leaves\': 100}"'))
@click.option('--holt_params',
              default='{}',
              help=('holt params. Must be a dict in a string format. ' +
                    'Example: --holt_params "{\'num_leaves\': 100}"'))
def model_training_script(arima_params, holt_params):
    """
    Generic Script that will train and log the models.
    """
    # Creating a dict to save results
    result={}

    # The destination folders must exist
    folders = ParamsValues.training_params['output']['folders']
    for f in folders:
        if not exists(f):
            makedirs(f)

    # Feature Pre-processing and selection:
    path = ParamsValues.training_params['input_train']
    logger.info(f'Reading data from {path}')

    # Read training data
    train = pd.read_csv(path)

    logger.info('Shape of dataset {}'.format(train.shape))

    path = ParamsValues.training_params['input_val']
    logger.info(f'Reading data from {path}')

    # Read training data
    test = pd.read_csv(path)
    # Preparing period
    test.period = pd.to_datetime(test.period).dt.to_period('M')
    test = test.set_index('period')

    arima_default_params = ParamsValues.training_params['model']['arima_params']
    if arima_params:
        arima_default_params.update(arima_params)

    holt_default_params = ParamsValues.training_params['model']['holt_params']
    if holt_params:
        holt_default_params.update(holt_params)

    logger.info(f"Executing the model Arima...")
    model_arima_training = model.ForecastingModel(arima_params=arima_default_params,
                                                       model_name='arima')
    model_arima_training.fit(train)
    prediction_a = model_arima_training.predict(12)
    metric_arima = model_arima_training.compute_metrics(test, prediction_a)
    
    result['arima']={'r2_score':metric_arima}
    
    logger.info(f"Executing the model Holt winter...")
    model_holt_training = model.ForecastingModel(holt_params=holt_default_params,
                                                      model_name='holt')
    model_holt_training.fit(train)
    prediction_h = model_holt_training.predict(12)
    metric_holt = model_holt_training.compute_metrics(test, prediction_h)
        
    result['holt']={'r2_score':metric_holt}
    
    if metric_holt > metric_arima:
       print(f'The best r2_score is Holt Winter: {round(metric_holt,2)}')
       result['best_model']=model_holt_training.model_name
       model_holt_training.save_model()
    else:
        print(f'The best r2_score is ARIMA: {round(metric_arima,2)}')
        result['best_model']=model_arima_training.model_name
        model_arima_training.save_model()
    
    #save results
    processed_dir = data_path('processed')
    with open(os.path.join(processed_dir, 'result.json'), 'w') as fp:
        json.dump(result, fp)

    #Persist predictions
    test['arima'] = prediction_a.values
    test['holt'] = prediction_h.values

    logger.info('Dumping prediction result data.')
    test.to_csv(os.path.join(processed_dir, 'predictions.csv'))

    logger.info('Dumping prediction result data figure.')
    test.index = test.index.astype(str)
    
    # Linear graph
    plt.figure(figsize=(20,10))
    plt.plot(test,
             label=test.columns)
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(processed_dir, 'predictions.jpg'))

if __name__ == '__main__':
    model_training_script()
