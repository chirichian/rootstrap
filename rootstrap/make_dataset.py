"""Run preprocess"""

import logging
import os
import click

from utils import data_path, load_data
from preprocessing.preprocess_pipe import Preprocessor

@click.command(name='make_dataset')
def make_dataset_cli():
    """Cleaning data and Make datasets from raw context files.

    Returns:
        pd.dataframe with the datasets processed and merged.
    """
    logger.info('Loading accounts data.')
    data = load_data()
    
    processed_dir = data_path('processed')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    logger.info('Preprocessing data pipeline.')
    processor = Preprocessor()
    train, test = processor.preprocess(data)
    
    logger.info(f'Dumping preprocessed data to: {processed_dir}')
    train.to_csv(os.path.join(
        processed_dir,'train_data.csv')
        )
    test.to_csv(
        os.path.join(processed_dir, 'test_data.csv')
        )
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    make_dataset_cli()
