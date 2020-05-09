import os
import traceback

from datasources.datasource import DatasourceFactory
from experiments.experiments import GenericExperiment
from nilmlab import exp_model_list
from nilmlab.factories import EnvironmentFactory
from nilmlab.lab import TimeSeriesLength
from nilmlab.exp_model_list import CLF_MODELS, TRANSFORMER_MODELS
from utils.logger import debug

dirname = os.path.dirname(__file__)
dirname = os.path.join(dirname, "../results")
if not os.path.exists(dirname):
    os.mkdir(dirname)
same_datasource_exp_checkpoint = os.path.join(dirname, '../results_from_generic_exp.csv')

appliances = ['microwave', 'dish washer', 'fridge', 'kettle', 'washer dryer',
              'toaster', 'television', 'hair dryer', 'vacuum cleaner']
env = EnvironmentFactory.create_env_single_building(datasource=DatasourceFactory.create_uk_dale_datasource(),
                                                    appliances=appliances)

experiment = GenericExperiment(env)

window = TimeSeriesLength.WINDOW_10_MINS
models = {}
if window == TimeSeriesLength.WINDOW_10_MINS:
    models = exp_model_list.selected_models_10mins
elif window == TimeSeriesLength.WINDOW_1_HOUR:
    models = exp_model_list.selected_models_1h
elif window == TimeSeriesLength.WINDOW_2_HOURS:
    models = exp_model_list.selected_models_2h
elif window == TimeSeriesLength.WINDOW_8_HOURS:
    models = exp_model_list.selected_models_8h
elif window == TimeSeriesLength.WINDOW_4_HOURS:
    models = exp_model_list.selected_models_4h
elif window == TimeSeriesLength.WINDOW_1_DAY:
    models = exp_model_list.selected_models_24h

for k in models.keys():
    experiment.setup_running_params(
        transformer_models=models[k][TRANSFORMER_MODELS],
        classifier_models=models[k][CLF_MODELS],
        train_appliances=appliances,
        test_appliances=appliances,
        ts_len=window,
        repeat=1)
    experiment.set_checkpoint_file(same_datasource_exp_checkpoint)
    tb = "No error"
    try:
        experiment.run()
    except Exception as e:
        tb = traceback.format_exc()
        debug(tb)
        debug(f"Failed for {k}")
        debug(f"{e}")
