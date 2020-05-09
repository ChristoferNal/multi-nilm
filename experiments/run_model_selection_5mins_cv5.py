import os

from experiments.experiments import ModelSelectionExperiment
from nilmlab import exp_model_list
from nilmlab.lab import TimeSeriesLength

dirname = os.path.dirname(__file__)
single_building_exp_checkpoint = os.path.join(dirname, '../results/cv5mins_ukdale1.csv')

exp = ModelSelectionExperiment()
exp.set_ts_len(TimeSeriesLength.WINDOW_5_MINS)
exp.set_checkpoint_file(single_building_exp_checkpoint)

exp.set_transformers(exp_model_list.model_selection_delay_embeddings)
exp.set_classifiers(exp_model_list.model_selection_rakel + exp_model_list.model_selection_mlknn)

exp.run()
