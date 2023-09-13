"""Predict"""

import click
import logging
import json
import os

from models.model import ForecastingModel
from utils import data_path

@click.command(name='predict')
@click.option('--n_periods',
              default=12)
def predict_cli(n_periods):
    """ Preprocess, build features and predict from scratch.

        Returns:
            pd.dataframe with the datasets processed and merged.
        """

    logger.info('Loading model.')
    processed_dir = data_path('processed')
    with open(os.path.join(processed_dir, 'result.json'), 'r') as fp:
        result = json.load(fp)
    model_name = result['best_model']
    model = ForecastingModel.import_model(model_name)

    y_predicted = model.predict(n_periods)

    print(y_predicted)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    predict_cli()
