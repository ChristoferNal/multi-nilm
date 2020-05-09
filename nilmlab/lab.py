import os
import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import List, Dict, Tuple, Union

import loguru
import numpy as np
import pandas as pd
from nilmtk import MeterGroup
from pandas import DataFrame
from sklearn.base import ClassifierMixin
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import cross_val_score

from datasources import labels_factory
from datasources.datasource import Datasource, SITE_METER
from nilmlab.lab_exceptions import NoSiteMeterException
from utils.logger import debug, info, timing


class TransformerType(Enum):
    # TODO: More clear TransformerType is needed.
    raw = 1
    transform = 2
    approximate = 3
    transform_and_approximate = 4


class TimeSeriesTransformer(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def transform(self, series: np.ndarray, sample_period: int = 6) -> list:
        """
        An interface to transform a given time series into another representation.
        It unifies different transformations and usually either just transforms a time series without dimensionality
        reduction or transforms a whole time series and reconstructs it using the underlying time series representation.
        Args:
            series (ndarray): A time series to be transformed according to the algorithm.
            sample_period (int): The sampling frequency.

        Returns:
            Returns the transformed time series as a list.
        """
        pass

    @abstractmethod
    def approximate(self, series: np.ndarray, window: int = 1, should_fit: bool = True) -> np.ndarray:
        """
        An interface to transform a given time series into another representation.
        In most transformers it transforms each segment of a time series, because the given time series is in segments.
        TODO: should_fit is used only by a few transformers. Move it to their constructors.
        Args:
            series (ndarray): A time series to be transformed according to the algorithm.
            window (int): The size of the sub-segments of the given time series.
                This is not supported by all algorithms.
            should_fit (bool): If the algorith should firstly fit to the data, executing some prepressing steps.
        Returns:
            Returns the transformed time series as ndarray.
        """
        pass

    @abstractmethod
    def reconstruct(self, series: np.ndarray) -> list:
        """
        It reconstructs the transformed time series.
        Args:
            series (ndarray): A transformed time series.

        Returns:
            The reconstructed time series as a list of values.
        """
        pass

    @abstractmethod
    def get_type(self) -> TransformerType:
        """
        Returns the type of the transformer, which indicates which functions the underlying algoirthm supports.
        Returns: A TransformerType.
        """
        pass

    @abstractmethod
    def set_type(self, method_type: TransformerType):
        """
        Sets the type of the transformer, which indicates which functions the underlying algoirthm supports.
        """
        pass

    @abstractmethod
    def get_name(self):
        pass

    def uses_labels(self):
        return False


def bucketize_data(data: np.ndarray, window: int) -> np.ndarray:
    """
    It segments the time series grouping it into batches. Its segment is of size equal to the window.
    Args:
        data (ndarray): The given time series.
        window (int): The size of the segments.

    Returns:

    """
    debug('bucketize_data: Initial shape {}'.format(data.shape))
    n_dims = len(data.shape)

    if n_dims == 1:
        seq_in_batches = np.reshape(data, (int(len(data) / window), window))
    elif n_dims == 2:
        seq_in_batches = np.reshape(data, (int(len(data) / window), window, data.shape[1]))
    else:
        raise Exception('Invalid number of dimensions {}.'.format(n_dims))
    debug('bucketize_data: Shape in batches: {}'.format(seq_in_batches.shape))
    return seq_in_batches


def bucketize_target(target: np.ndarray, window: int) -> np.ndarray:
    """
    Creates target data according to the lenght of the window of the segmented data.
    Args:
        target (ndarray): Target data with the original size.
        window (int): The length of window that will be used to create the corresponding labels.
    Returns:
        The target data for the new bucketized time series.
    """
    target_in_batches = bucketize_data(target, window)
    any_multilabel = np.any(target_in_batches, axis=1)
    debug('bucketize_target: Shape of array in windows: {}'.format(target_in_batches.shape))
    debug('bucketize_target: Shape of array after merging windows: {}'.format(any_multilabel.shape))
    return any_multilabel


class TimeSeriesLength(Enum):
    """
    The length of each segment of the time series, which will be used for inference.
    """
    WINDOW_SAMPLE_PERIOD = 'same'
    WINDOW_1_MIN = '1m'
    WINDOW_5_MINS = '5m'
    WINDOW_10_MINS = '10m'
    WINDOW_30_MINS = '30m'
    WINDOW_1_HOUR = '1h'
    WINDOW_2_HOURS = '2h'
    WINDOW_4_HOURS = '4h'
    WINDOW_8_HOURS = '8h'
    WINDOW_1_DAY = '1d'
    WINDOW_1_WEEK = '1w'


def repeat_the_same_date_for_all_buildings(buildings: List[int], end_date: List[str], start_date: List[str]):
    """
    Creates a list of start and end dates for as many buildings as in the given list of buildings.
    Args:
        buildings (List[int]): The given buildings that will be used.
        end_date (List[str]): End date of the data that will be selected for each building.
        start_date (List[str]): Start date of the data that will be selected for each building.
    """
    for i in range(1, len(buildings)):
        start_date.append(start_date[-1])
        end_date.append(end_date[-1])


def dates_as_lists(end_date: Union[str, List[str]], start_date: Union[str, List[str]]):
    """
    If the given dates are pure strings convert them to lists of strings.
    Args:
        end_date (Union[str, List[str]]): End date of the data that will be selected for each building.
        start_date (List[str]): Start date of the data that will be selected for each building.
    Returns:
        Returns the start and end dates as lists of dates.
    """
    if not isinstance(start_date, list):
        start_date = [start_date]
    if not isinstance(end_date, list):
        end_date = [end_date]
    return end_date, start_date


class Environment:
    """
    This class describes all the parameters related to the data.
    """

    def __init__(self, datasource: Datasource,
                 buildings: Union[int, List[int]],
                 year: Union[str, str],
                 start_date: Union[str, List[str]],
                 end_date: Union[str, List[str]],
                 sample_period: int = 6,
                 appliances: List = None,
                 is_deep_classifier=False):
        """
        Constructs a new Environment with the given parameters.
        Args:
            datasource (Datasource): The data source that will be used to load energy data.
            buildings (Union[int, List[int]]): The given buildings that will be used.
            year (Union[str, str]): The year or the range of years that are used. This parameter doesn't affect the
                actual experiments, it is used mainly as a summary of the period of the data of the created environment.
            start_date (Union[str, List[str]]): Start date of the data that will be selected for each building.
            end_date (Union[str, List[str]]): End date of the data that will be selected for each building.
            sample_period (int): The sampling frequency.
            appliances (List): A list of appliances.
            is_deep_classifier (bool): This is a flag that is used in case of deep neural networks.
        """
        self.datasource = datasource
        self.buildings = buildings
        self.year = year
        self.start_date = start_date
        self.end_date = end_date
        self.sample_period = sample_period
        self.appliances = appliances
        self.is_deep_classifier = is_deep_classifier
        if not buildings:
            raise EnvironmentError("Building is not specified.")
        if isinstance(buildings, int) or len(buildings) == 1:
            if isinstance(buildings, int):
                building = buildings
            else:
                building = buildings[0]
            if isinstance(start_date, list):
                start_date = start_date[0]
            if isinstance(end_date, list):
                end_date = end_date[0]
            all_df, metergroup, label2id = self.setup_one_building(appliances, datasource, building,
                                                                   start_date, end_date, sample_period)
            labels_df = labels_factory.create_multilabels_from_meters(all_df, metergroup, label2id)
        else:
            end_date, start_date = dates_as_lists(end_date, start_date)
            if isinstance(start_date, list) and isinstance(end_date, list):

                if len(start_date) == len(end_date) and len(start_date) == 1 and len(buildings) > 1:
                    repeat_the_same_date_for_all_buildings(buildings, end_date, start_date)

                if len(start_date) != len(buildings) or len(end_date) != len(buildings):
                    raise EnvironmentError("Number of buildings not the same with number of dates")

            buildings_with_dates = list(zip(buildings, start_date, end_date))
            data_per_building: Dict[int, Tuple[DataFrame, MeterGroup, Dict]] = \
                self.setup_across_many_buildings(appliances, datasource, buildings_with_dates, sample_period)
            labels_per_building = labels_factory.create_multilabels_from_many_buildings(data_per_building)

            data_frame = []
            labels_frame = []
            metergroup = None
            for building in data_per_building.keys():
                df, metergroup_of_building, label2id = data_per_building[building]
                if not metergroup:
                    metergroup = metergroup_of_building
                else:
                    metergroup = metergroup.union(metergroup_of_building)
                data_frame.append(df)
                labels_frame.append(labels_per_building[building])
            all_df = pd.concat(data_frame)
            labels_df = pd.concat(labels_frame)

        self.all_df, self.metergroup, self.labels_df = all_df, metergroup, labels_df
        self.ts_transformer = None
        self.multilabel_clf = None

        self.train_datasource = datasource
        self.train_building = buildings
        self.train_year = year
        self.train_start_date = start_date
        self.train_end_date = end_date
        self.train_sample_period = sample_period
        self.train_df = self.all_df
        self.train_labels_df = self.labels_df
        # self.train_label2id = self.label2id

        self.test_datasource = None
        self.test_building = None
        self.test_year = None
        self.test_start_date = None
        self.test_end_date = None
        self.test_sample_period = None
        self.test_df = None
        self.test_labels_df = None
        self.test_label2id = None
        self.ts_length = TimeSeriesLength.WINDOW_SAMPLE_PERIOD

    def setup_train_data(self, datasource: Datasource = None,
                         building: int = None,
                         year: str = None,
                         start_date: str = None,
                         end_date: str = None,
                         sample_period: int = 6,
                         appliances: List = None):
        """
        Setup training data.
        Args:
            datasource (Datasource): The Datasource that will be used for training.
            building (int): The building that will be used for training.
            year (str): The year(s) that the training data correspond to.
            start_date (str): Start date of the data that will be selected for each building.
            end_date (str): End date of the data that will be selected for each building.
            sample_period (int): The sampling frequency.
            appliances (List): A list of appliances.
        """
        if datasource is not None:
            self.train_datasource = datasource
        if building is not None:
            self.train_building = building
        if year is not None:
            self.train_year = year
        if start_date is not None:
            self.train_start_date = start_date
        if end_date is not None:
            self.train_end_date = end_date
        if sample_period is not None:
            self.train_sample_period = sample_period
        if appliances:
            self.appliances = appliances
        self.train_df, train_metergroup, train_label2id = self.setup_one_building(appliances, datasource, building,
                                                                                  start_date, end_date, sample_period)
        self.train_labels_df = labels_factory.create_multilabels_from_meters(self.train_df,
                                                                             train_metergroup,
                                                                             train_label2id)

    def setup_test_data(self, datasource: Datasource = None,
                        building: int = None,
                        year: str = None,
                        start_date: str = None,
                        end_date: str = None,
                        sample_period: int = 6,
                        appliances: List = None):
        """
        Setup the testing data.
        Args:
            datasource (Datasource): The Datasource that will be used for testing.
            building (int): The building that will be used for testing.
            year (str): The year(s) that the testing data correspond to.
            start_date (str): Start date of the data that will be selected for each building.
            end_date (str): End date of the data that will be selected for each building.
            sample_period (int): The sampling frequency.
            appliances (List): A list of appliances.
        """
        if datasource is not None:
            self.test_datasource = datasource
        if building is not None:
            self.test_building = building
        if year is not None:
            self.test_year = year
        if start_date is not None:
            self.test_start_date = start_date
        if end_date is not None:
            self.test_end_date = end_date
        if sample_period is not None:
            self.test_sample_period = sample_period
        if appliances:
            self.appliances = appliances
        self.test_df, test_metergroup, test_label2id = self.setup_one_building(appliances, datasource, building,
                                                                               start_date, end_date, sample_period)
        self.test_labels_df = labels_factory.create_multilabels_from_meters(self.test_df,
                                                                            test_metergroup,
                                                                            test_label2id)

    def set_deep_classifier(self, is_deep_clf: bool = True):
        """
        Set to true if a deep neural network is used as a classifier.
        Args:
            is_deep_clf (bool):

        Returns:

        """
        self.is_deep_classifier = is_deep_clf

    def set_ts_len(self, ts_length: TimeSeriesLength):
        """
        Set the length of the segments of the given time series.
        Args:
            ts_length (TimeSeriesLength): The length of the segments of the time series.
        """
        self.ts_length = ts_length

    def get_ts_len(self) -> TimeSeriesLength:
        """
        It returns the length of the segments of the time series.
        Returns: A TimeSeriesLength that corresponds to the size of the segments of the time series.
        """
        return self.ts_length

    def get_multilabels(self, labels_df: DataFrame, appliances: List = None) -> DataFrame:
        """
        Get the labels of the specified appliances.
        Args:
            labels_df (DataFrame):
            appliances (List):

        Returns:

        """
        debug(f"get_multilabels  labels_df.columns {labels_df.columns}")
        debug(f"get_multilabels  appliances {appliances}")
        if appliances is None:
            return labels_df
        else:
            return labels_df[appliances]

    def get_site_meter_data(self, df: DataFrame) -> np.ndarray:
        """
        Get the data of the site meter from the given DataFrame.
        Args:
            df (DataFrame): A DataFrame containing energy data with columns corresponding to different meters.

        Returns:
            The site meter data as an array (ndarray).
        """
        for col in df.columns:
            if SITE_METER in col:
                return df[col].values
        raise NoSiteMeterException("Couldn' t find site meter.")

    def get_window(self, dt: TimeSeriesLength) -> int:
        """
        Get the number of samples that correspond to the given TimeSeriesLength.
        The result may vary depending on the sampling rate that is predefined.
        Args:
            dt (TimeSeriesLength): The given TimeSeriesLength in time.
        Returns:
            The number of samples that correspond to the time length.
        """
        choices = {TimeSeriesLength.WINDOW_SAMPLE_PERIOD: 1,
                   TimeSeriesLength.WINDOW_1_MIN        : self.get_no_of_samples_per_min(),
                   TimeSeriesLength.WINDOW_5_MINS       : self.get_no_of_samples_per_min() * 5,
                   TimeSeriesLength.WINDOW_10_MINS      : self.get_no_of_samples_per_min() * 10,
                   TimeSeriesLength.WINDOW_30_MINS      : self.get_no_of_samples_per_min() * 30,
                   TimeSeriesLength.WINDOW_1_HOUR       : self.get_no_of_samples_per_hour(),
                   TimeSeriesLength.WINDOW_2_HOURS      : self.get_no_of_samples_per_hour() * 2,
                   TimeSeriesLength.WINDOW_4_HOURS      : self.get_no_of_samples_per_hour() * 4,
                   TimeSeriesLength.WINDOW_8_HOURS      : self.get_no_of_samples_per_hour() * 8,
                   TimeSeriesLength.WINDOW_1_DAY        : self.get_no_of_samples_per_day(),
                   TimeSeriesLength.WINDOW_1_WEEK       : self.get_no_of_samples_per_day() * 7
                   }
        return int(choices.get(dt, 1))

    def get_features(self, data_df: DataFrame, representation: TransformerType = TransformerType.raw) -> List:
        """
        It transforms the given data using underlying algorithm that is wrapped by the TimeSeriesTransformer interface.
        Args:
            data_df (DataFrame): The time series that will be transformed into another time series representation.
            representation (TransformerType): The type of transformation that the specified TimeSeriesTransformer
             supports.
        Returns:
            A list containing the converted time series.
        """
        data = self.get_site_meter_data(data_df)
        if representation == TransformerType.transform or representation == TransformerType.transform_and_approximate:
            if self.ts_transformer is None:
                raise Exception('TimeSeriesTransformer has not been placed!')
            data = self.ts_transformer.transform(data)
        return data

    def reduce_dimensions(self, data_in_batches: np.ndarray, window: int, target: np.ndarray, should_fit: bool = True):
        """
        It uses the method approximate of the TimeSeriesTransformer in order to achieve dimensionality reduction.
        Args:
            data_in_batches (ndarray): The data of the time series separated in batches.
            window (int): The size of the sub-segments of the given time series.
                This is not supported by all algorithms.
            target (ndarray): The labels that correspond to the given data in batches.
            should_fit (bool): True if it is supported by the algorithm of the specified time series representation.
        Returns:
            The shortened time series as an array (ndarray).

        """
        if self.ts_transformer is None:
            raise Exception('TimeSeriesTransformer has not been placed!')
        if self.ts_transformer.uses_labels():
            squeezed_seq = self.ts_transformer.approximate(data_in_batches, window, target, should_fit)
        else:
            squeezed_seq = self.ts_transformer.approximate(data_in_batches, window, should_fit=should_fit)

        debug('Shape of squeezed seq: {}'.format(squeezed_seq.shape))
        return squeezed_seq

    def get_no_of_samples_per_min(self):
        """
        It returns the number of samples per minute. This depends also on the predefined sample period.
        Returns:
            An int representing the number of samples.
        """
        return 60 / self.sample_period

    def get_no_of_samples_per_hour(self):
        """
        It returns the number of samples per hour. This depends also on the predefined sample period.
        Returns:
            An int representing the number of samples.
        """
        return self.get_no_of_samples_per_min() * 60

    def get_no_of_samples_per_day(self):
        """
        It returns the number of samples per day. This depends also on the predefined sample period.
        Returns:
            An int representing the number of samples.
        """
        return self.get_no_of_samples_per_hour() * 24

    def place_ts_transformer(self, transformer: TimeSeriesTransformer):
        """
        Set the time series transformer that will be used.
        Args:
            transformer (TimeSeriesTransformer): The time series transformer that will be used.
        """
        self.ts_transformer = transformer

    def place_multilabel_classifier(self, multilabel_clf: Union[str, ClassifierMixin]):
        """
        Specify the multi label classifier that will be used.
        Args:
            multilabel_clf (Union[str, ClassifierMixin]):
        """
        if isinstance(multilabel_clf, str):
            self.is_deep_classifier = True
            self.multilabel_clf = multilabel_clf
        else:
            self.multilabel_clf = multilabel_clf

    def setup_across_many_buildings(self, appliances, datasource, buildings_with_dates: List[Tuple[int, str, str]],
                                    sample_period: int) -> Dict[int, Tuple[DataFrame, MeterGroup, Dict]]:
        """
        Setup using many buildings.
        Args:
            appliances (List): The appliances that will be recongized.
            datasource (Datasource): The Datasource that will be used to load energy data.
            buildings_with_dates (List[Tuple[int, str, str]]): The dates for all the buildings that will be used.
            sample_period (int): The sampling frequency.

        Returns:
            A dictionary containing the loaded data for each building.
        """
        data_per_building = dict()
        for building, start_date, end_date in buildings_with_dates:
            loguru.logger.info(f"setup across many buildings: building {building}, start {start_date}, end {end_date}")
            df, metergroup, label2id = self.setup_one_building(appliances, datasource, building,
                                                               start_date, end_date, sample_period)
            data_per_building[building] = (df, metergroup, label2id)
        return data_per_building

    @staticmethod
    def setup_one_building(appliances, datasource, building, start_date, end_date,
                           sample_period) -> (pd.DataFrame, MeterGroup, Dict, Dict):
        """
        Setup and load the data using one building.
        Args:
            appliances (List): The appliances that will be recongized.
            datasource (Datasource): The Datasource that will be used to load energy data.
            building (int): The building that is used.
            start_date (str): Start date of the data that will be selected for each building.
            end_date (str): End date of the data that will be selected for each building.
            sample_period (int): The sampling frequency.
        Returns:

        """
        if appliances:
            info(f'Reading data from specified meters. \n-Building: {building}\n-Appliances {appliances}')
            all_df, metergroup = datasource.read_selected_appliances(appliances=appliances, start=start_date,
                                                                     end=end_date,
                                                                     sample_period=sample_period, building=building)
        else:
            info('Reading data from all meters...')
            all_df, metergroup = datasource.read_all_meters(start_date, end_date,
                                                            building=building,
                                                            sample_period=sample_period)

        loguru.logger.debug(f"Length of data of all loaded meters {len(all_df)}")
        all_df, label2id = datasource.normalize_columns(all_df, metergroup, appliances)
        loguru.logger.debug(f"Length of data of all loaded meters {len(all_df)}")
        info('Meters that have been loaded (all_df.columns):\n' + str(all_df.columns))

        return all_df, metergroup, label2id

    def get_type_of_transformer(self) -> TransformerType:
        """
        Get the type of the transformer.
        Returns: The type of the transformer (TransformerType)
        """
        if self.ts_transformer is None:
            raise Exception('TimeSeriesTransformer has not been placed!')
        return self.ts_transformer.get_type()

    def cross_validate(self, appliances: list, cv: int = 5,
                       raw_data: bool = False):
        """
        Execute a cross validation.
        Args:
            appliances (List): List of appliances to be recognized.
            cv (int): The number sets to be used for cross validation.
            raw_data (bool): If the experiment uses raw data without any time series representation.
        Returns:
            A tuple with macro and micro f scores. Currently micro is disabled and returns 0.
                    scores : array of float, shape=(len(list(cv)),)
                    Array of scores of the estimator for each run of the cross validation.
        """
        # TODO: Define overlap for windows
        # TODO: define the case window=1
        # TODO: Clarify TS_TRASFORMATION and TS_APPROXIMATION cases
        ts_length = self.get_ts_len()
        data, target = self._preprocess(self.all_df, self.labels_df, appliances, ts_length, raw_data)

        if len(data.shape) == 3:
            data = np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2]))

        debug(f"Unique classes {target}")
        macro_scores = cross_val_score(self.multilabel_clf, data, target, cv=cv, scoring='f1_macro', n_jobs=-1)
        info('F1 macro: {} (+/- {})'.format(macro_scores.mean(), macro_scores.std()))

        # micro_scores = cross_val_score(self.multilabel_clf, data, target, cv=cv, scoring='f1_micro', n_jobs=-1)
        # info('F1 micro: {} (+/- {})'.format(micro_scores.mean(), micro_scores.std()))
        micro_scores = np.array([0, 0])
        return macro_scores, micro_scores

    def train(self, appliances: list, raw_data: bool = False):
        """
        Train the algorithm for the specified appliances.
        Args:
            appliances (List): List of appliances to be recognized.
            raw_data (bool): True if the experiment uses raw data without any time series representation.
        Returns:
            The preprocess and the fiting time.
        """
        info("Prepossessing before training...")
        start_time = time.time()
        data, target = self._preprocess(self.train_df, self.train_labels_df, appliances, self.get_ts_len(), raw_data)
        preprocess_time = time.time() - start_time
        timing(f"preprocess time {preprocess_time}")

        if len(data.shape) == 3:
            data = np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2]))

        info("Training...")
        start_time = time.time()
        self.multilabel_clf.fit(data, target)
        fit_time = time.time() - start_time
        timing(f"fit time {fit_time}")
        return preprocess_time, fit_time

    def test(self, appliances: list, raw_data: bool = False):
        """
        Runs a test using the specified appliances.
        Args:
            appliances (List): List of appliances to be recognized.
            raw_data (bool): True if the experiment uses raw data without any time series representation.
        Returns:
            A tuple containing macro, micro, a report, preprocess and fiting time.
        """
        if self.test_df is None or self.test_labels_df is None:
            raise (Exception('Test data or test target is None'))
        info("Prepossessing before testing...")
        start_time = time.time()
        data, target = self._preprocess(self.test_df, self.test_labels_df, appliances,
                                        self.get_ts_len(), raw_data, should_fit=False)
        preprocess_time = time.time() - start_time
        timing(f"preprocess time {preprocess_time}")
        if len(data.shape) == 3:
            data = np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2]))
        info("Testing...")

        start_time = time.time()
        predictions = self.multilabel_clf.predict(data)
        predictions_time = time.time() - start_time
        timing(f"predictions time {predictions_time}")

        micro = f1_score(target, predictions, average='micro')
        macro = f1_score(target, predictions, average='macro')
        info('F1 macro {}'.format(macro))
        info('F1 micro {}'.format(micro))
        report = classification_report(target, predictions, target_names=appliances, output_dict=True)
        # confusion_matrix = multilabel_confusion_matrix(y_true=target, y_pred=predictions.toarray())
        # confusion_matrix = None
        return macro, micro, report, preprocess_time, predictions_time

    def _preprocess(self, data_df, labels_df, appliances, ts_length, raw_data, should_fit: bool = True):
        if self.multilabel_clf is None:
            raise Exception('Multilabel classifier has not been placed!')
        if raw_data:
            representation_type = TransformerType.raw
        else:
            representation_type = self.get_type_of_transformer()
        debug(f"Type of transformer {representation_type}")

        start_time = time.time()
        data = self.get_features(data_df, representation_type)
        get_features_time = time.time() - start_time
        timing(f"get features time {get_features_time}")

        debug(f"Features \n {data[:10]}")
        target = self.get_multilabels(labels_df, appliances)
        target = np.array(target.values)
        debug(f"Target \n {target[:10]}")
        window = self.get_window(ts_length)
        rem = len(data) % window
        if rem > 0:
            data = data[:-rem]
            target = target[:-rem]
        target = bucketize_target(target, window)
        data = bucketize_data(data, window)
        # if representation_type == TransformerType.raw or representation_type == TransformerType.approximate:
        #     pass
        if representation_type == TransformerType.approximate \
                or representation_type == TransformerType.transform_and_approximate:
            start_time = time.time()
            data = self.reduce_dimensions(data, window, target, should_fit)
            reduce_dimensions_time = time.time() - start_time
            timing(f"reduce dimensions time {reduce_dimensions_time}")

        return data, target


class Experiment(ABC):
    """
    Abstract class describing an multi label disaggregation experiment.
    """
    columns_results = [
        'train_end_date',
        'train_start_date',
        'train_sample_period',
        'train_building',
        'train_datasource',
        'test_end_date',
        'test_start_date',
        'test_sample_period',
        'test_building',
        'test_datasource',
        'ts length',
        'classifier',
        'clf properties',
        'ts_representation',
        'transformer_type',
        'ts_repr properties',
        'cross validation',
        'macro avg',
        'macro sd',
        'micro avg',
        'micro sd',
        'num_of_appliances',
        'appliances'
    ]

    def __init__(self):
        super().__init__()
        self.env = None
        self.train_end_date = None
        self.train_start_date = None
        self.train_sample_period = None
        self.train_building = None
        self.train_datasource_name = None

        self.test_end_date = None
        self.test_start_date = None
        self.test_sample_period = None
        self.test_building = None
        self.test_datasource_name = None
        self.ts_length = None
        self.transformers = None
        self.classifiers = None
        self.deep_classifiers = None

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def setup_environment(self):
        pass

    def set_transformers(self, transformers: List):
        self.transformers = transformers

    def set_classifiers(self, classifiers: List):
        self.classifiers = classifiers

    def set_deep_classifiers(self, classifiers: List[str]):
        self.deep_classifiers = classifiers

    def populate_environment(self, environment: Environment):
        self.env = environment

    def populate_ts_params(self):
        self.ts_length = self.env.get_ts_len()

    def populate_train_parameters(self, env: Environment):
        self.train_end_date = env.train_end_date
        self.train_start_date = env.train_start_date
        self.train_sample_period = env.train_sample_period
        self.train_building = env.train_building
        self.train_datasource_name = env.train_datasource.get_name()

    def populate_test_parameters(self):
        if self.env.test_datasource is None:
            return
        self.test_end_date = self.env.test_end_date
        self.test_start_date = self.env.test_start_date
        self.test_sample_period = self.env.test_sample_period
        self.test_building = self.env.test_building
        self.test_datasource_name = self.env.test_datasource.get_name()

    def create_description(self,
                           classifier: str,
                           clf_properties: str,
                           ts_representation: str,
                           transformer_type: str,
                           ts_repr_properties: str,
                           cross_validation: str,
                           macro_avg: str,
                           macro_sd: str,
                           micro_avg: str,
                           micro_sd: str,
                           num_of_appliances: str,
                           appliances: str,
                           report: str = None,
                           training_time: str = None,
                           testing_time: str = None,
                           preprocess_time: str = None,
                           prediction_time: str = None,
                           preprocess_train_time: str = None,
                           fit_time: str = None) -> dict:
        self.populate_ts_params()
        self.populate_test_parameters()
        debug(f"train building {self.train_building}")
        description = {
            'train_end_date'       : str(self.train_end_date),
            'train_start_date'     : str(self.train_start_date),
            'train_sample_period'  : str(self.train_sample_period),
            'train_building'       : str(self.train_building),
            'train_datasource'     : str(self.train_datasource_name),
            'test_end_date'        : str(self.test_end_date),
            'test_start_date'      : str(self.test_start_date),
            'test_sample_period'   : str(self.test_sample_period),
            'test_building'        : str(self.test_building),
            'test_datasource'      : str(self.test_datasource_name),
            'ts length'            : str(self.ts_length),
            'classifier'           : classifier,
            'clf_properties'       : clf_properties,
            'ts_representation'    : ts_representation,
            'transformer_type'     : transformer_type,
            'ts_repr_properties'   : ts_repr_properties,
            'cross_validation'     : cross_validation,
            'macro_avg'            : macro_avg,
            'macro_sd'             : macro_sd,
            'micro_avg'            : micro_avg,
            'micro_sd'             : micro_sd,
            'num_of_appliances'    : num_of_appliances,
            'appliances'           : appliances,
            'report'               : report,
            'training_time'        : training_time,
            'testing_time'         : testing_time,
            'preprocess_time'      : preprocess_time,
            'prediction_time'      : prediction_time,
            'preprocess_train_time': preprocess_train_time,
            'fit_time'             : fit_time
        }
        return description

    def save_experiment(self, description, reset_results, results_file):
        new_results_df = pd.DataFrame(description, index=[0])
        results_csv = Path(results_file)
        if reset_results and results_csv.is_file():
            os.remove(results_file)
        if results_csv.is_file():
            results_df = pd.read_csv(results_csv)
            results_df = results_df.append(new_results_df)
        else:
            results_df = new_results_df
        results_df.to_csv(results_csv, index=False)
        info(str(results_df.tail()))
