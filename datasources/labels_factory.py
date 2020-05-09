import time
from typing import Dict, Tuple

import loguru
from nilmtk import MeterGroup
from numba import njit
import numpy as np
from pandas import DataFrame

from datasources.datasource import SITE_METER
from utils.logger import TIMING, debug, timing


def create_multilabels_from_meters(meters: DataFrame, meter_group: MeterGroup, labels2id: dict) -> DataFrame:
    """
    Creates multi labels from the given meter group using a dictionary as a lookup table.
    Args:
        meters (DataFrame):
        meter_group (MeterGroup):
        labels2id (dict):

    Returns:
        A DataFrame with the multi labels.
    """
    start_time = time.time() if TIMING else None
    labels = dict()
    for col in meters.columns:
        loguru.logger.info(f"Creating multilabels from meter {col}, "
                           f"\nlabels2id[col] {labels2id[col]}"
                           f"\nmetergroup[labels2id[col]] {meter_group[labels2id[col]]}")
        meter = meter_group[labels2id[col]]
        threshold = meter.on_power_threshold()
        vals = meters[col].values.astype(float)
        if vals is None or col == SITE_METER:
            loguru.logger.debug(f"Skipping {col} - {vals}")
            continue
        loguru.logger.debug(f"meters[col].values.astype(float) {col} - {vals}")
        labels[col] = create_labels(vals, threshold)
    timing('Create multilabels from meters {}'.format(round(time.time() - start_time, 2)))
    return DataFrame(labels)


def create_multilabels_from_many_buildings(data_per_building: Dict[int, Tuple[DataFrame, MeterGroup, Dict]]) \
        -> Dict[int, DataFrame]:
    """
    Creates multi labels given more than one buildings.
    Args:
        data_per_building (Dict[int, Tuple[DataFrame, MeterGroup, Dict]]): A dictionary with keys the numbers of the buildings and values tuples containing
        the necessary data to create labels by calling create_multilabels_from_meters()

    Returns:
        A dictionary with the labels for each building.
    """
    labels_per_building = dict()
    for building in data_per_building.keys():
        df, metergroup, label2id = data_per_building[building]
        labels_df = create_multilabels_from_meters(df, metergroup, label2id)
        labels_per_building[building] = labels_df
    return labels_per_building


def create_multilabels(appliances: dict, meter_group: MeterGroup) -> dict:
    """
        Creates labels from the given meter group for the given appliances.
    Args:
        appliances (dict): dict with keys ['oven', 'microwave', 'dish washer', 'fridge freezer', 'kettle', 'washer dryer',
              'toaster', 'boiler', 'television', 'hair dryer', 'vacuum cleaner', 'light']

        meter_group (MeterGroup): A MeterGroup object.

    Returns:
        A dictionary with labels per meter.
    """
    start_time = time.time() if TIMING else None
    labels = dict()

    for key in appliances.keys():
        meter = meter_group.submeters()[key]
        threshold = meter.on_power_threshold()
        labels[meter.label() + str(meter.instance())] = create_labels(appliances[key], threshold)
        debug('{} threshold = {}'.format(meter.label(), threshold))

    timing('Create multilabels {}'.format(round(time.time() - start_time, 2)))
    return labels


@DeprecationWarning
def apply_create_labels_on_df(dataframe, threshold):
    """This method is 300 times slower than create_labels which uses numba"""
    return dataframe.apply(lambda x: 1 if x >= threshold else 0)


@njit(parallel=True)
def create_labels(array, threshold):
    res = np.empty(array.shape)
    for i in range(len(array)):
        if array[i] >= threshold:
            res[i] = 1
        else:
            res[i] = 0
    return list(res)
