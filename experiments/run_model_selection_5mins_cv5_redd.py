import os

from experiments.experiments import REDDModelSelectionExperiment

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from nilmlab import exp_model_list
from nilmlab.lab import TimeSeriesLength

dirname = os.path.dirname(__file__)
single_building_exp_checkpoint = os.path.join(dirname, '../results/cv5mins_redd3.csv')

exp = REDDModelSelectionExperiment(building=3)
exp.set_ts_len(TimeSeriesLength.WINDOW_5_MINS)
exp.set_checkpoint_file(single_building_exp_checkpoint)
exp.set_transformers(exp_model_list.cv_signal2vec)
exp.set_classifiers(exp_model_list.cv_signal2vec_clf)
exp.run()