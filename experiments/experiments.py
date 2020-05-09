import time
from typing import Tuple, List, Any

from datasources.datasource import DatasourceFactory
from nilmlab import exp_model_list
from nilmlab.lab import Environment, Experiment, TimeSeriesLength
from utils.logger import timing

reset_results = False


class ModelSelectionExperiment(Experiment):
    appliances = ['oven', 'microwave', 'dish washer', 'fridge freezer', 'kettle', 'washer dryer',
                  'toaster', 'boiler', 'television', 'hair dryer', 'vacuum cleaner', 'light']
    results_file: str
    ts_len: TimeSeriesLength

    def __init__(self, cv=3):
        super().__init__()
        self.transformers = exp_model_list.model_selection_transformers
        self.classifiers = exp_model_list.model_selection_clf_list
        self.cv = cv

    def setup_environment(self):
        train_year = '2014'
        train_month_end = '8'
        train_month_start = '1'
        train_end_date = "{}-30-{}".format(train_month_end, train_year)
        train_start_date = "{}-1-{}".format(train_month_start, train_year)
        train_sample_period = 6
        train_building = 1
        train_datasource = DatasourceFactory.create_uk_dale_datasource()

        test_year = '2014'
        test_month_end = '9'
        test_month_start = '7'
        test_end_date = "{}-30-{}".format(test_month_end, test_year)
        test_start_date = "{}-1-{}".format(test_month_start, test_year)
        test_sample_period = 6
        test_building = 1
        test_datasource = DatasourceFactory.create_uk_dale_datasource()
        env = Environment(train_datasource, train_building, train_year, train_start_date, train_end_date,
                          train_sample_period, self.appliances)
        self.populate_environment(env)
        self.populate_train_parameters(env)

    def run(self):
        self.setup_environment()
        self.env.set_ts_len(self.ts_len)

        for transformer in self.transformers:
            for clf in self.classifiers:
                self.env.place_multilabel_classifier(clf)
                self.env.place_ts_transformer(transformer)
                macro_scores, micro_scores = self.env.cross_validate(self.appliances, cv=self.cv, raw_data=False)

                description = self.create_description(type(clf).__name__,
                                                      str(clf),
                                                      transformer.get_name(),
                                                      str(self.env.get_type_of_transformer()),
                                                      str(transformer),
                                                      str(self.cv),
                                                      macro_scores.mean(),
                                                      macro_scores.std(),
                                                      micro_scores.mean(),
                                                      micro_scores.std(),
                                                      str(len(self.appliances)),
                                                      str(self.appliances))

                self.save_experiment(description, reset_results, self.results_file)

    def set_checkpoint_file(self, results_file: str = '../results/cross_val_window_4_hours.csv'):
        self.results_file = results_file

    def set_ts_len(self, ts_len: TimeSeriesLength = TimeSeriesLength.WINDOW_4_HOURS):
        self.ts_len = ts_len


class GenericExperiment(Experiment):
    results_file: str
    ts_len: TimeSeriesLength

    def __init__(self, environment):
        super().__init__()
        self.env = environment
        self.transformers = None
        self.classifiers = None
        self.train_appliances = []
        self.test_appliances = []
        self.repeat = 1

    def setup_environment(self):
        self.env.set_ts_len(self.ts_len)
        self.populate_environment(self.env)
        self.populate_train_parameters(self.env)

    def setup_running_params(self,
                             transformer_models: List[Tuple[Any, str]],
                             classifier_models: List[Tuple[Any, str]],
                             train_appliances,
                             test_appliances=None,
                             ts_len: TimeSeriesLength = TimeSeriesLength.WINDOW_4_HOURS,
                             repeat: int = 1):
        self.set_transfomers_and_classifiers(transformer_models, classifier_models)
        self.set_ts_len(ts_len)
        self.repeat = repeat
        self.train_appliances = train_appliances
        if test_appliances:
            self.test_appliances = test_appliances
        else:
            self.test_appliances = train_appliances

    def run(self):
        self.setup_environment()
        if len(self.transformers) != len(self.classifiers):
            raise Exception("List of transformers doesn't have the same length with list of classifiers. "
                            "It should be a 1-1 map")

        for model_index in range(len(self.transformers)):
            transformer = self.transformers[model_index]
            transformer_descr = str(transformer)
            clf = self.classifiers[model_index]
            clf_descr = str(clf)
            for i in range(self.repeat):
                self.env.place_multilabel_classifier(clf)
                self.env.place_ts_transformer(transformer)
                start_time = time.time()
                preprocess_train_time, fit_time = self.env.train(self.train_appliances)
                training_time = time.time() - start_time
                timing(f"training time {training_time}")
                start_time = time.time()
                macro, micro, report, preprocess_time, prediction_time = self.env.test(self.test_appliances)
                testing_time = time.time() - start_time
                timing(f"testing time {testing_time}")

                description = self.create_description(type(clf).__name__,
                                                      clf_descr,
                                                      transformer.get_name(),
                                                      str(self.env.get_type_of_transformer()),
                                                      transformer_descr,
                                                      "train/test",
                                                      macro,
                                                      None,
                                                      micro,
                                                      None,
                                                      str(len(self.train_appliances)),
                                                      str(self.train_appliances),
                                                      str(report),
                                                      str(training_time),
                                                      str(testing_time),
                                                      str(preprocess_time),
                                                      str(prediction_time),
                                                      str(preprocess_train_time),
                                                      str(fit_time)
                                                      )

                self.save_experiment(description, reset_results, self.results_file)

    def set_checkpoint_file(self, results_file: str = '../results/cross_val_window_4_hours.csv'):
        self.results_file = results_file

    def set_ts_len(self, ts_len: TimeSeriesLength = TimeSeriesLength.WINDOW_4_HOURS):
        self.ts_len = ts_len

    def set_transfomers_and_classifiers(self, transformer_models: List[Tuple[Any, str]],
                                        classifier_models: List[Tuple[Any, str]]):
        self.transformers = transformer_models
        self.classifiers = classifier_models

    def set(self, environment):
        self.env = environment


class REDDModelSelectionExperiment(ModelSelectionExperiment):
    appliances_redd3 = ['electric furnace', 'CE appliance', 'microwave', 'washer dryer', 'unknown', 'sockets']
    appliances_redd1 = ['electric oven', 'fridge', 'microwave', 'washer dryer', 'unknown', 'sockets', 'light']

    results_file: str
    ts_len: TimeSeriesLength

    def __init__(self, building=1, cv=2):
        super().__init__()
        self.transformers = exp_model_list.model_selection_transformers
        self.classifiers = exp_model_list.model_selection_clf_list
        self.building = building
        self.cv = cv

    def setup_environment(self):
        train_year = '2011'
        train_month_end = '5'
        train_month_start = '4'
        train_end_date = "{}-30-{}".format(train_month_end, train_year)
        train_start_date = "{}-1-{}".format(train_month_start, train_year)
        train_sample_period = 6
        train_building = self.building
        if self.building == 1:
            self.appliances = self.appliances_redd1
        elif self.building == 3:
            self.appliances = self.appliances_redd3

        train_datasource = DatasourceFactory.create_redd_datasource()

        env = Environment(train_datasource, train_building, train_year, train_start_date, train_end_date,
                          train_sample_period, self.appliances)
        self.populate_environment(env)
        self.populate_train_parameters(env)
