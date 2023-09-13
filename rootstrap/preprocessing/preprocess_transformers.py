"""Transformers for loading and cleaning dataset"""
import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from kb import ParamsValues


logger = logging.getLogger(__name__)


class passengersTransformer(BaseEstimator, TransformerMixin):
    """Passenger transformer."""
    def __init__(self, 
                filename='International_Report_Passengers.csv',
                is_predict=None,
                columns_name=['Total', 'period']):
        """
        Args:
            columns_name: list, column's name that you will work with
        """
        self.columns_names = columns_name
        self.filename = filename
        self.is_predict = is_predict

    def fit(self, X, y=None):
        """Fit method.

        Args:
            X: pd.DataFrame.

        Returns:
            self.
        """
        return self

    @staticmethod    
    def get_periods(X):
        """
            Get the period columns from datetime mm/dd/yyyy
        Args:
            X pd.DataFrame with column data_dte
        Return:
            pd.DataFrame with period as a new column
        """

        # Converting date_dte as date format
        X['data_dte'] = pd.to_datetime(X.data_dte)
        # Getting period
        X['period'] = X.data_dte.dt.to_period('M')

        return X
        
    @staticmethod
    def filter_periods(X, periods_key):
        """Filter non wanted periods

        Args:
            X: pd.DataFrame with column period.

        Returns:
            Filtered pd.DataFrame.
        """

        if "Year" not in X.columns:
            raise ValueError("Period column not in DataFrame")

        PERIODS = ParamsValues.features_static_values.get(periods_key)

        logger.info('Filtering periods')

        before = X.shape[0]
        X = X[X.Year.isin(PERIODS)]
        after = X.shape[0]

        logger.info(f'Before total rows: {before:,d}')
        logger.info(f'After total rows: {after:,d}')
        logger.info(f'Dropped {before-after:,d} rows')

        return X

    def transform(self, X):
        """Transform input to compute median and max tmv per account

        Args:
            X: pd.DataFrame with columns: 'data_dte', 'Year','Total'

        Returns:
            Filtered and cleaned pd.DataFrame
        """
        
        if not all([c in X.columns for c in ['data_dte', 'Year','Total']]):
            raise ValueError("Missing column in DataFrame")

        logger.info('Processing data')

        if not self.is_predict:
            periods_key = 'valid_modeling_years'
        else:
            periods_key = 'predict_periods'

        X = self.get_periods(X)
        X = self.filter_periods(X, periods_key)

        # Reset index after filtering the DataFrame
        X = X.reset_index(drop=True)

        time_series = X.groupby('period').Total.sum()

        return time_series

    def get_feature_names(self):
        return self.columns_names



class passengersSplitter(BaseEstimator, TransformerMixin):
    """Passenger splitter."""
    def __init__(self, 
                train_size=None,
                test_size=12):
        """
        Args:
            test_size: int, number of month to use in trainig
        """
        self.test_size = test_size
        self.train_size = train_size

    def fit(self, X, y=None):
        """Fit method.

        Args:
            X: pd.DataFrame.

        Returns:
            self.
        """
        return self

    def transform(self, X):
        """Transform method.

        Args:
            X: pd.DataFrame.

        Returns:
            train, test: pd.DataFrame
        """
        X = X.sort_index()
        test = X[-self.test_size:]
        train = X.drop(test.index)

        return train, test