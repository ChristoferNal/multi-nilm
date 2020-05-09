import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from fuzzywuzzy import fuzz
from nilmtk import DataSet, MeterGroup
from pandas import DataFrame

from datasources.paths_manager import UK_DALE, REDD
from nilmlab.lab_exceptions import LabelNormalizationError
from utils.logger import timing, TIMING, info, debug

NAME_UK_DALE = 'UK DALE'
NAME_REDD = 'REDD'
SITE_METER = 'Site meter'


class Datasource():

    def __init__(self, dataset: DataSet, name: str):
        self.dataset = dataset
        self.name = name

    def get_dataset(self):
        return self.dataset

    def get_name(self):
        return self.name

    def read_all_meters(self, start: str, end: str, sample_period: int = 6, building: int = 1) \
            -> Tuple[DataFrame, MeterGroup]:
        """
        Read the records during the given start and end dates, for all the meters of the given building.
        Args:
            start (str): The starting date in the format "{month}-{day of month}-{year}" e.g. "05-30-2012".
            end (str): The final date in the format "{month}-{day of month}-{year}" e.g. "08-30-2012".
            sample_period (int): The sample period of the records.
            building (int): The building to read the records from.

        Returns:
            Returns a tuple containing the respective DataFrame and MeterGroup of the data that are read.
        """
        start_time = time.time() if TIMING else None
        self.dataset.set_window(start=start, end=end)
        elec = self.dataset.buildings[building].elec
        timing('NILMTK selecting all meters: {}'.format(round(time.time() - start_time, 2)))

        start_time = time.time() if TIMING else None
        df = elec.dataframe_of_meters(sample_period=sample_period)
        timing('NILMTK converting all meters to dataframe: {}'.format(round(time.time() - start_time, 2)))

        df.fillna(0, inplace=True)
        return df, elec

    def read_selected_appliances(self, appliances: List, start: str, end: str, sample_period=6, building=1,
                                 include_mains=True) -> Tuple[DataFrame, MeterGroup]:
        """
        Loads the data of the specified appliances.
        Args:
            appliances (List): A list of appliances to read their records.
            start (str): The starting date in the format "{month}-{day of month}-{year}" e.g. "05-30-2012".
            end (str): The final date in the format "{month}-{day of month}-{year}" e.g. "08-30-2012".
            sample_period (int): The sample period of the records.
            building (int): The building to read the records from.
            include_mains (bool): True if should include main meters.

        Returns:
            Returns a tuple containing the respective DataFrame and MeterGroup of the data that are read.
        """
        debug(f" read_selected_appliances {appliances}, {building}, {start}, {end}, {include_mains}")

        selected_metergroup = self.get_selected_metergroup(appliances, building, end, start, include_mains)

        start_time = time.time() if TIMING else None
        df = selected_metergroup.dataframe_of_meters(sample_period=sample_period)
        timing('NILMTK converting specified appliances to dataframe: {}'.format(round(time.time() - start_time, 2)))

        debug(f"Length of data of read_selected_appliances {len(df)}")
        df.fillna(0, inplace=True)
        return df, selected_metergroup

    def read_mains(self, start, end, sample_period=6, building=1) -> Tuple[DataFrame, MeterGroup]:
        """
        Loads the data of the specified appliances.
        Args:
            start (str): The starting date in the format "{month}-{day of month}-{year}" e.g. "05-30-2012".
            end (str): The final date in the format "{month}-{day of month}-{year}" e.g. "08-30-2012".
            sample_period (int): The sample period of the records.
            building (int): The building to read the records from.

        Returns:
            Returns a tuple containing the respective DataFrame and MeterGroup of the data that are read.
        """
        self.dataset.set_window(start=start, end=end)
        mains_meter = self.dataset.buildings[building].elec.mains()
        if isinstance(mains_meter, MeterGroup):
            mains_metergroup = mains_meter
        else:
            mains_metergroup = MeterGroup(meters=[mains_meter])
        start_time = time.time() if TIMING else None
        df = mains_metergroup.dataframe_of_meters(sample_period=sample_period)
        timing('NILMTK converting mains to dataframe: {}'.format(round(time.time() - start_time, 2)))

        df.fillna(0, inplace=True)
        return df, mains_metergroup

    # def read_data_of_appliance(self, start, end, sample_period=6, building=1, device=None) -> np.ndarray:
    #     """
    #     Reads the data of a specific appliance. If no device is specified then it reads the main meter.
    #     :param start:
    #     :type start:
    #     :param end:
    #     :type end:
    #     :param sample_period:
    #     :type sample_period:
    #     :param building:
    #     :type building:
    #     :param device:
    #     :type device:
    #     :return:
    #     :rtype:
    #     """
    #     start_time = time.time() if TIMING else None
    #
    #     power_df = self.read_df(start, end, sample_period, building, device)
    #     power_data = power_df.values
    #
    #     debug('Power data shape {}'.format(power_data.shape))
    #     debug('Type of power_data {}'.format(type(power_data)))
    #     debug('Size of power_data {}'.format(len(power_data)))
    #     timing('NILMTK reading and getting power series: {}'.format(round(time.time() - start_time, 2)))
    #     self.clean_nans(power_data)
    #
    #     return power_data

    def get_selected_metergroup(self, appliances, building, end, start, include_mains) -> MeterGroup:
        """
        Gets a MeterGroup with the specified appliances for the given building during the given dates.
        Args:
            appliances (List): A list of appliances to read their records.
            building (int): The building to read the records from.
            start (str): The starting date in the format "{month}-{day of month}-{year}" e.g. "05-30-2012".
            end (str): The final date in the format "{month}-{day of month}-{year}" e.g. "08-30-2012".
            include_mains (bool): True if should include main meters.

        Returns:
            A MeterGroup containing the specified appliances.
        """
        start_time = time.time() if TIMING else None
        self.dataset.set_window(start=start, end=end)
        elec = self.dataset.buildings[building].elec
        appliances_with_one_meter = []
        appliances_with_more_meters = []
        for appliance in appliances:
            metergroup = elec.select_using_appliances(type=appliances)
            if len(metergroup.meters) > 1:
                appliances_with_more_meters.append(appliance)
            else:
                appliances_with_one_meter.append(appliance)

        special_metergroup = None
        for appliance in appliances_with_more_meters:
            inst = 1
            if appliance == 'sockets' and building == 3:
                inst = 4
            if special_metergroup is None:
                special_metergroup = elec.select_using_appliances(type=appliance, instance=inst)
            else:
                special_metergroup = special_metergroup.union(elec.select_using_appliances(type=appliance, instance=1))

        selected_metergroup = elec.select_using_appliances(type=appliances_with_one_meter)
        selected_metergroup = selected_metergroup.union(special_metergroup)
        if include_mains:
            mains_meter = self.dataset.buildings[building].elec.mains()
            if isinstance(mains_meter, MeterGroup):
                if len(mains_meter.meters) > 1:
                    mains_meter = mains_meter.meters[0]
                    mains_metergroup = MeterGroup(meters=[mains_meter])
                else:
                    mains_metergroup = mains_meter
            else:
                mains_metergroup = MeterGroup(meters=[mains_meter])
            selected_metergroup = selected_metergroup.union(mains_metergroup)
        timing('NILMTK select using appliances: {}'.format(round(time.time() - start_time, 2)))
        return selected_metergroup

    @staticmethod
    def normalize_columns(df: DataFrame, meter_group: MeterGroup, appliance_names: List[str]) -> Tuple[DataFrame, dict]:
        """
        It normalizes the names of the columns for compatibility.
        Args:
            df (DataFrame):
            meter_group (MeterGroup):
            appliance_names (List[str]):

        Returns:
            A tuple with a DataFrame and a dictionary mapping labels to ids.
        """
        labels = meter_group.get_labels(df.columns)
        normalized_labels = []
        info(f"Df columns before normalization {df.columns}")
        info(f"Labels before normalization {labels}")

        for label in labels:
            if label == SITE_METER and SITE_METER not in appliance_names:
                normalized_labels.append(SITE_METER)
                continue
            for name in appliance_names:
                ratio = fuzz.ratio(label.lower().replace('electric', "").lstrip().rstrip().split()[0],
                                   name.lower().replace('electric', "").lstrip().rstrip().split()[0])
                if ratio > 90:
                    info(f"{name} ~ {label} ({ratio}%)")
                    normalized_labels.append(name)
        if len(normalized_labels) != len(labels):
            debug(f"len(normalized_labels) {len(normalized_labels)} != len(labels) {len(labels)}")
            raise LabelNormalizationError()
        label2id = {l: i for l, i in zip(normalized_labels, df.columns)}
        df.columns = normalized_labels
        info(f"Normalized labels {normalized_labels}")
        return df, label2id

    @staticmethod
    def rename_columns(df: DataFrame, meter_group: MeterGroup) -> (DataFrame, dict, dict):
        """
        Rename columns of the given DataFrame using the respective labels of each meter.
        Args:
            df (DataFrame):
            meter_group (MeterGroup):

        Returns:
            Returns a DataFrame with renamed columns and two dictionaries to covnert labels to ids and vice versa.
        """
        new_columns = []
        label2id = dict()
        id2label = dict()
        for col in df.columns:
            try:
                meter = meter_group[col]
                label = meter.label() + str(col[0])
                new_columns.append(label)
                label2id[label] = col
                id2label[col] = label
            except KeyError:
                info(f"KeyError key={col}")
        df.columns = new_columns
        return df, label2id, id2label

    # def read_df(self, start, end, sample_period=6, building=1, device=None):
    #     self.dataset.set_window(start=start, end=end)
    #     elec = self.dataset.buildings[building].elec
    #     if device is not None:
    #         mains = elec.submeters()[device]
    #         debug('Reading data of {}.'.format(device))
    #     else:
    #         mains = elec.mains()
    #         debug('Reading data of mains.')
    #     power_df = mains.power_series_all_data(sample_period=sample_period)
    #     return power_df

    @staticmethod
    def clean_nans(data):
        start_time = time.time() if TIMING else None
        np.nan_to_num(data, False)
        timing('None to num: {}'.format(round(time.time() - start_time, 2)))


class DatasourceFactory:
    """
    It is responsible to create different data sources that are based on various data sets.
    """

    @staticmethod
    def create_uk_dale_datasource():
        return Datasource(DatasourceFactory.get_uk_dale_dataset(), NAME_UK_DALE)

    @staticmethod
    def get_uk_dale_dataset():
        return DataSet(UK_DALE)

    @staticmethod
    def create_redd_datasource():
        return Datasource(DatasourceFactory.get_redd_dataset(), NAME_REDD)

    @staticmethod
    def get_redd_dataset():
        return DataSet(REDD)


def save_and_plot(sequence, plot=False, save_figure=False, filename=None):
    if plot or save_figure:
        plt.plot(sequence)
        if filename is not None and save_figure:
            plt.savefig(filename + '.png')
        if plot:
            plt.show()
