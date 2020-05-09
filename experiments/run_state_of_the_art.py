"""

Model selection environment parameters
- UKDALE Building 1   1/1/2014 - 30/6/2014
- REDD   Building 1,3 1/4/2011 - 30/5/2011
Train-test environment parameters
- UKDALE Building 1
  Train: 1/3/2013 - 30/6/2014
  Test : 1/7/2014 - 31/12/2014
- REDD   Building 1
  Train: 18/4/2011 - 17/5/2011
  Test : 18/5/2011 - 25/5/2011
- REDD   Building 3
  Train: 16/4/2011 - 30/4/2011
  Test : 17/5/2011 - 30/5/2011

  redd 3
  2011-04-16 01:11:24-04:00 - 2011-05-30 20:19:54-04:00

  redd 1
  2011-04-18 09:22:06-04:00 - 2011-05-24 15:57:00-04:00
"""

import os
import traceback

from datasources.datasource import DatasourceFactory
from experiments.experiments import GenericExperiment
from nilmlab import exp_model_list
from nilmlab.factories import EnvironmentFactory
from nilmlab.lab import TimeSeriesLength
from nilmlab.exp_model_list import CLF_MODELS, TRANSFORMER_MODELS, BOSS, SIGNAL2VEC, TIME_DELAY_EMBEDDING
from utils.logger import debug

dirname = os.path.dirname(__file__)

STATE_OF_THE_ART = os.path.join(dirname, '../results/state_of_the_art_performance.csv')
APPLIANCES_UK_DALE_BUILDING_1 = ['oven', 'microwave', 'dish washer', 'fridge freezer',
                                 'kettle', 'washer dryer', 'toaster', 'boiler', 'television',
                                 'hair dryer', 'vacuum cleaner', 'light']
APPLIANCES_REDD_BUILDING_1 = ['electric oven', 'fridge', 'microwave', 'washer dryer', 'unknown', 'sockets', 'light']
APPLIANCES_REDD_BUILDING_3 = ['electric furnace', 'CE appliance', 'microwave', 'washer dryer', 'unknown', 'sockets']

ukdale_train_year_start = '2013'
ukdale_train_year_end = '2014'
ukdale_train_month_end = '5'
ukdale_train_month_start = '3'
ukdale_train_end_date = "{}-30-{}".format(ukdale_train_month_end, ukdale_train_year_end)
ukdale_train_start_date = "{}-1-{}".format(ukdale_train_month_start, ukdale_train_year_start)

ukdale_test_year_start = '2014'
ukdale_test_year_end = '2014'
ukdale_test_month_end = '12'
ukdale_test_month_start = '6'
ukdale_test_end_date = "{}-30-{}".format(ukdale_test_month_end, ukdale_test_year_end)
ukdale_test_start_date = "{}-1-{}".format(ukdale_test_month_start, ukdale_test_year_start)

redd1_train_year_start = '2011'
redd1_train_year_end = '2011'
redd1_train_month_end = '5'
redd1_train_month_start = '4'
redd1_train_end_date = "{}-17-{}".format(redd1_train_month_end, redd1_train_year_end)
redd1_train_start_date = "{}-18-{}".format(redd1_train_month_start, redd1_train_year_start)

redd1_test_year_start = '2011'
redd1_test_year_end = '2011'
redd1_test_month_end = '5'
redd1_test_month_start = '5'
redd1_test_end_date = "{}-25-{}".format(redd1_test_month_end, redd1_test_year_end)
redd1_test_start_date = "{}-18-{}".format(redd1_test_month_start, redd1_test_year_start)

redd3_train_year_start = '2011'
redd3_train_year_end = '2011'
redd3_train_month_end = '4'
redd3_train_month_start = '4'
redd3_train_end_date = "{}-30-{}".format(redd3_train_month_end, redd3_train_year_end)
redd3_train_start_date = "{}-16-{}".format(redd3_train_month_start, redd3_train_year_start)

redd3_test_year_start = '2011'
redd3_test_year_end = '2011'
redd3_test_month_end = '5'
redd3_test_month_start = '5'
redd3_test_end_date = "{}-30-{}".format(redd1_test_month_end, redd1_test_year_end)
redd3_test_start_date = "{}-17-{}".format(redd1_test_month_start, redd1_test_year_start)

env_ukdale_building_1 = EnvironmentFactory.create_env_single_building(
    datasource=DatasourceFactory.create_uk_dale_datasource(),
    building=1,
    sample_period=6,
    train_year=ukdale_train_year_start + "-" + ukdale_train_year_end,
    train_start_date=ukdale_train_start_date,
    train_end_date=ukdale_train_end_date,
    test_year=ukdale_test_year_start + "-" + ukdale_test_year_end,
    test_start_date=ukdale_test_start_date,
    test_end_date=ukdale_test_end_date,
    appliances=APPLIANCES_UK_DALE_BUILDING_1)

ukdale_building1_experiment = GenericExperiment(env_ukdale_building_1)

env_redd_building_1 = EnvironmentFactory.create_env_single_building(
    datasource=DatasourceFactory.create_redd_datasource(),
    building=1,
    sample_period=6,
    train_year=redd1_train_year_start + "-" + redd1_train_year_end,
    train_start_date=redd1_train_start_date,
    train_end_date=redd1_train_end_date,
    test_year=redd1_test_year_start + "-" + redd1_test_year_end,
    test_start_date=redd1_test_start_date,
    test_end_date=redd1_test_end_date,
    appliances=APPLIANCES_REDD_BUILDING_1)

redd_building1_experiment = GenericExperiment(env_redd_building_1)

ukdale_building3_experiment = GenericExperiment(env_ukdale_building_1)

env_redd_building_3 = EnvironmentFactory.create_env_single_building(
    datasource=DatasourceFactory.create_redd_datasource(),
    building=3,
    sample_period=6,
    train_year=redd3_train_year_start + "-" + redd3_train_year_end,
    train_start_date=redd3_train_start_date,
    train_end_date=redd3_train_end_date,
    test_year=redd3_test_year_start + "-" + redd3_test_year_end,
    test_start_date=redd3_test_start_date,
    test_end_date=redd3_test_end_date,
    appliances=APPLIANCES_REDD_BUILDING_3)

redd_building3_experiment = GenericExperiment(env_redd_building_3)


def run_experiments(experiment, appliances, window):
    models = exp_model_list.state_of_the_art
    for k in models.keys():
        experiment.setup_running_params(
            transformer_models=models[k][TRANSFORMER_MODELS],
            classifier_models=models[k][CLF_MODELS],
            train_appliances=appliances,
            test_appliances=appliances,
            ts_len=window,
            repeat=1)
        experiment.set_checkpoint_file(STATE_OF_THE_ART)
        tb = "No error"
        try:
            experiment.run()
        except Exception as e:
            tb = traceback.format_exc()
            debug(tb)
            debug(f"Failed for {k}")
            debug(f"{e}")


run_experiments(ukdale_building1_experiment, APPLIANCES_UK_DALE_BUILDING_1, TimeSeriesLength.WINDOW_5_MINS)
run_experiments(redd_building1_experiment, APPLIANCES_REDD_BUILDING_1, TimeSeriesLength.WINDOW_10_MINS)
run_experiments(redd_building1_experiment, APPLIANCES_REDD_BUILDING_1, TimeSeriesLength.WINDOW_30_MINS)
run_experiments(redd_building1_experiment, APPLIANCES_REDD_BUILDING_1, TimeSeriesLength.WINDOW_1_HOUR)
run_experiments(redd_building1_experiment, APPLIANCES_REDD_BUILDING_1, TimeSeriesLength.WINDOW_2_HOURS)
run_experiments(redd_building1_experiment, APPLIANCES_REDD_BUILDING_1, TimeSeriesLength.WINDOW_4_HOURS)
run_experiments(redd_building1_experiment, APPLIANCES_REDD_BUILDING_1, TimeSeriesLength.WINDOW_8_HOURS)
run_experiments(redd_building1_experiment, APPLIANCES_REDD_BUILDING_1, TimeSeriesLength.WINDOW_1_DAY)
