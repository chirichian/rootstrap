"""Create pipeline"""

import logging
from sklearn.pipeline import make_pipeline

from preprocessing.preprocess_transformers import (
    passengersSplitter,
    passengersTransformer
    )

logger = logging.getLogger(__name__)

class Preprocessor():
    """Preprocessor module."""
    def __init__(self, model_name=None, skip_init=None):
        """Init method.

        Args:
            model_name: str
        """
        self.model_name = model_name
        if not skip_init:
            self.__set_preprocess_pipeline()

    def __set_preprocess_pipeline(self):
        """Set preprocess pipeline."""
        pass_transformer = passengersTransformer()
        pass_splitter = passengersSplitter()

        self.preprocess_pipe = make_pipeline(
            pass_transformer,
            pass_splitter,
        )

    def preprocess(self, data):
        """Preprocess features and targets.

        Args:
            data: pandas dataframe.
        """

        train, test = self.preprocess_pipe.transform(data)

        return train, test

   