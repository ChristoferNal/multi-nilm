import os

from experiments.experiments import ModelSelectionExperiment
from nilmlab.lab import TimeSeriesLength

dirname = os.path.dirname(__file__)
single_building_exp_checkpoint = os.path.join(dirname, '../results/cv1h.csv')

exp = ModelSelectionExperiment(cv=5)
exp.set_ts_len(TimeSeriesLength.WINDOW_1_HOUR)
exp.set_checkpoint_file(single_building_exp_checkpoint)
exp.run()