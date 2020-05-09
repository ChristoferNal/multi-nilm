from typing import List
from typing import Union

from pyts import approximation, transformation
from tslearn import piecewise
from tslearn.piecewise import SymbolicAggregateApproximation, OneD_SymbolicAggregateApproximation

from datasources.datasource import DatasourceFactory, Datasource
from nilmlab.lab import Environment
from nilmlab.lab import TransformerType
from nilmlab.tstransformers import TSLearnTransformerWrapper, PytsTransformerWrapper, Signal2Vec, WaveletAdapter, \
    TimeDelayEmbeddingAdapter

SECONDS_PER_DAY = 60 * 60 * 24

CAPACITY15GB = 1024 * 1024 * 1024 * 15

reset_results = False


class EnvironmentFactory:

    @staticmethod
    def create_env_single_building(datasource: Datasource,
                                   building: int = 1,
                                   sample_period: int = 6,
                                   train_year: str = "2013-2014",
                                   train_start_date: str = "3-1-2013",
                                   train_end_date: str = "5-30-2014",
                                   test_year: str = "2014",
                                   test_start_date: str = "6-1-2014",
                                   test_end_date: str = "12-30-2014",
                                   appliances: List = None):
        """
        The specific experiment includes training and testing on house 1 of UK-DALE.
        The test set is defined as the year following April 2016, while the rest of the data are available
        for training.
        """
        env = Environment(datasource, building, train_year, train_start_date, train_end_date, sample_period, appliances)
        env.setup_test_data(datasource=datasource, building=building, year=test_year,
                            start_date=test_start_date, end_date=test_end_date, appliances=appliances)
        return env

    @staticmethod
    def create_env_single_building_learning_and_generalization_on_the_same_dataset(
            datasource: Datasource,
            sample_period: int = 6,
            train_building: int = 1,
            train_year: str = '2013-2014',
            train_start_date: str = '3-1-2013',
            train_end_date: str = '5-30-2014',
            test_building: int = 2,
            test_year: str = '2014',
            test_start_date: str = '6-1-2014',
            test_end_date: str = '12-30-2014',
            appliances: List = None):
        """
        House 1 of UK-DALE is selected as the training set here again, while the rest of the
        houses where the target appliance is present compose the test sets.
        If an appliance is not present in the training or test building an error will be thrown.
        """
        env = Environment(datasource, train_building, train_year,
                          train_start_date, train_end_date, sample_period, appliances)
        env.setup_test_data(datasource=datasource, building=test_building, year=test_year,
                            start_date=test_start_date, end_date=test_end_date, appliances=appliances)
        return env

    @staticmethod
    def create_env_multi_building_learning_and_generalization_on_the_same_dataset(
            datasource: Datasource = DatasourceFactory.create_uk_dale_datasource(),
            sample_period: int = 6,
            train_building: Union[int, List[int]] = (1, 2),
            train_year: str = '2013-2014',
            train_start_date: str = '3-1-2013',
            train_end_date: str = '5-30-2014',
            test_building: int = 5,
            test_year: str = '2014',
            test_start_date: str = '6-1-2014',
            test_end_date: str = '12-30-2014',
            appliances: List = ('fridge', 'microwave')):
        """
        The experiments used for this category are defined for the UK-DALE dataset.
        """
        env = Environment(datasource, train_building, train_year,
                          train_start_date, train_end_date, sample_period, appliances)
        env.setup_test_data(datasource=datasource, building=test_building, year=test_year,
                            start_date=test_start_date, end_date=test_end_date, appliances=appliances)
        return env

    @staticmethod
    def create_env_generalization_on_different_dataset(train_datasource: Datasource,
                                                       sample_period: int,
                                                       train_building: int,
                                                       train_year: str,
                                                       train_start_date: str,
                                                       train_end_date: str,
                                                       test_datasource: Datasource,
                                                       test_building: int,
                                                       test_year: str,
                                                       test_start_date: str,
                                                       test_end_date: str):
        """
        The training set is comprised of UK-DALE data, while testing is applied to REDD data.
        The first has buildings in the UK, while the second is for buildings in USA.
        """
        env = Environment(train_datasource, train_building, train_year, train_start_date, train_end_date, sample_period)
        env.setup_test_data(datasource=test_datasource, building=test_building, year=test_year,
                            start_date=test_start_date, end_date=test_end_date)
        return env


class TransformerFactory:

    @staticmethod
    def build_tslearn_paa(n_paa_segments=50, supports_approximation=True):
        paa = piecewise.PiecewiseAggregateApproximation(n_paa_segments)
        return TSLearnTransformerWrapper(paa, supports_approximation=supports_approximation)

    @staticmethod
    def build_tslearn_sax(n_paa_segments=50, n_sax_symbols=50, supports_approximation=True):
        sax = SymbolicAggregateApproximation(n_segments=n_paa_segments,
                                             alphabet_size_avg=n_sax_symbols)
        return TSLearnTransformerWrapper(sax, supports_approximation=supports_approximation)

    @staticmethod
    def build_tslearn_one_d_sax(n_paa_segments=50, n_sax_symbols=50):
        one_d_sax = OneD_SymbolicAggregateApproximation(n_segments=n_paa_segments,
                                                        alphabet_size_avg=n_sax_symbols,
                                                        alphabet_size_slope=4)
        return TSLearnTransformerWrapper(one_d_sax, supports_approximation=False)

    @staticmethod
    def build_pyts_paa(n_paa_segments=50):
        paa = approximation.PiecewiseAggregateApproximation(window_size=None, output_size=n_paa_segments,
                                                            overlapping=False)
        return PytsTransformerWrapper(paa)

    @staticmethod
    def build_pyts_sax(n_sax_symbols=50):
        sax = approximation.SymbolicAggregateApproximation(n_bins=n_sax_symbols,
                                                           alphabet=[i for i in range(n_sax_symbols)])
        return PytsTransformerWrapper(sax)

    @staticmethod
    def build_pyts_dft(n_coefs=30, norm_mean=False, norm_std=False, supports_approximation=True):
        dft = approximation.DiscreteFourierTransform(n_coefs=n_coefs, norm_mean=norm_mean, norm_std=norm_std)
        return PytsTransformerWrapper(dft, supports_approximation)

    @staticmethod
    def build_pyts_sfa(n_coefs=50, n_bins=5, norm_mean=False, norm_std=False):
        sfa = approximation.SymbolicFourierApproximation(n_coefs=n_coefs,
                                                         norm_mean=norm_mean,
                                                         norm_std=norm_std,
                                                         n_bins=n_bins,
                                                         alphabet=[i for i in range(n_bins)]
                                                         )
        return PytsTransformerWrapper(sfa)

    @staticmethod
    def build_pyts_boss(word_size=2, n_bins=5, window_size=10, norm_mean=False, norm_std=False):
        # TODO: Check other parameters
        boss = transformation.BOSS(word_size=word_size,
                                   window_size=window_size,
                                   norm_mean=norm_mean,
                                   norm_std=norm_std,
                                   n_bins=n_bins
                                   )
        return PytsTransformerWrapper(boss)

    @staticmethod
    def build_pyts_weasel(word_size=2, n_bins=4, norm_mean=False, norm_std=False):
        # TODO: Check other parameters
        weasel = transformation.WEASEL(word_size=word_size,
                                       norm_mean=norm_mean,
                                       norm_std=norm_std,
                                       n_bins=n_bins
                                       )
        return PytsTransformerWrapper(weasel)

    @staticmethod
    def build_signal2vec(classifier_path: str, embedding_path: str,
                         transformer_type: TransformerType = TransformerType.transform_and_approximate,
                         num_of_vectors: int = 1):
        signal2vec = Signal2Vec(classifier_path, embedding_path, num_of_representative_vectors=num_of_vectors)
        signal2vec.set_type(transformer_type)
        return signal2vec

    @staticmethod
    def build_wavelet(wavelet_name: str = 'haar', filter_bank: str = None, mode='symmetric', level=None, drop_cA=False,
                      transformer_type: TransformerType = TransformerType.approximate):
        wavelet_adapter = WaveletAdapter(wavelet_name=wavelet_name, filter_bank=filter_bank, mode=mode,
                                         level=level, drop_cA=drop_cA)
        wavelet_adapter.set_type(transformer_type)
        return wavelet_adapter

    @staticmethod
    def build_delay_embedding(delay_in_seconds: int, dimension: int, sample_period: int = 6,
                              transformer_type: TransformerType = TransformerType.approximate):
        wavelet_adapter = TimeDelayEmbeddingAdapter(delay_in_seconds=delay_in_seconds, dimension=dimension,
                                                    sample_period=sample_period)
        wavelet_adapter.set_type(transformer_type)
        return wavelet_adapter
