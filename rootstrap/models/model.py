"""Train and test predictive churn model"""


import datetime
import joblib
import pmdarima as pmd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
from sklearn.metrics import r2_score

from kb import ParamsValues

import logging
logger = logging.getLogger(__name__)


class ForecastingModel():
    """
    This class is in charge of train,
    validate and save the statistical model
    """
    def __init__(self, 
                 model_name,
                 arima_params=None, 
                 holt_params=None, 
                 training_version=None):
        """ Run the complete experiment.
        Args:
            model_name: str, options are arima, holt
            arima_params: dict. Values to be used in model
            holt_params: dict. Values to be used in model
            training_version: Str. Model version name
        """
        self.model_name = model_name
        self.final_test_metrics = None
        self.model = None
        self.params = {
            'arima_params': {},
            'holt_params': {},
        }
        # TODO: Add output paths parameters in init class
        self.outputs = {
            'arima_model': ParamsValues.training_params['output']['arima_model'],
            'holt_model': ParamsValues.training_params['output']['holt_model'],
            'best_model' : ParamsValues.predict_params['best_model']
        }

        if arima_params:
            self.params['arima_params'].update(arima_params)
        
        if holt_params:
            self.params['holt_params'].update(holt_params)

        if training_version:
            self.training_version = training_version
        else:
            self.training_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def fit(self, train):
        """
        Train model, and save results

        Args:
            train: pd.DataFrame
            test: pd.DataFrame
            model: string, model to train, options holt, arima
        Returns:
            None
        """

        # Model Training
        if self.model_name=='arima':
            train.period = pd.to_datetime(train.period).dt.to_period('M')
            train = train.set_index('period')
            self.model = pmd.auto_arima(
                train, **self.params['arima_params']
                )
        if self.model_name=='holt':
            train.period = train.period.astype(str)
            train = train.set_index('period')
            self.model = ExponentialSmoothing(
                train, **self.params['holt_params']
                ).fit()


    def compute_metrics(self, y_true, y_pred):
        """
        Compute metrics

        Args:
            y_true:
            y_pred:
        Returns:
            None
        """

        # Save metrics
        self.final_test_metrics = {
            'r2_score': r2_score(y_true, y_pred),

        }
        return r2_score(y_true, y_pred)

    def predict(self, n_periods):
        """
        Compute predictions for X_test dataset

        Args:
            X_test: pd.DataFrame
            y_test: (optional). pd.DataFrame
        Returns:
            array-like with predicted values
        """
        if not self.model:
            # TODO: Raise exception
            print('Have to train a model first')
            pass
        if self.model_name=='arima':
            y_pred = self.model.predict(n_periods=n_periods)
        if self.model_name=='holt':
            y_pred = self.model.forecast(n_periods)
        return y_pred

    
    def save_model(self, artifact_path=None):
        """
        Save model object to a file.
        """
    
        if self.model_name=='arima':
            artifact_path = self.outputs['arima_model']
            joblib.dump(value=self.model, filename=artifact_path)

        if self.model_name=='holt':
            artifact_path = self.outputs['holt_model']
            joblib.dump(value=self.model, filename=artifact_path)


    def __load_model(self, artifact_path=None):
        """
        Load model object to class variable
        """

        if not artifact_path:
            artifact_path = self.outputs['best_model']
        self.model = joblib.load(filename=artifact_path)
        
    @classmethod
    def import_model(cls,model_name):
        """
        Import model and return ModelTraining class
        """

        model_training = cls(model_name)
        model_training.__load_model()

        return model_training

