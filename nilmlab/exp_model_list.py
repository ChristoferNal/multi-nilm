from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import ExtraTreeClassifier
from skmultilearn.adapt import MLkNN
from skmultilearn.ensemble import RakelD

from datasources.paths_manager import SAVED_MODEL, PATH_SIGNAL2VEC
from nilmlab.factories import TransformerFactory
from nilmlab.lab import TransformerType

SAX = 'SAX'
SAX1D = 'SAX1D'
SFA = 'SFA'
DFT = 'DFT'
PAA = 'PAA'
WEASEL = 'WEASEL'
SIGNAL2VEC = 'SIGNAL2VEC'
TRANSFORMER_MODELS = 'TRANSFORMER_MODELS'
CLF_MODELS = 'CLF_MODELS'
BOSS = 'BOSS'
TIME_DELAY_EMBEDDING = 'TIME_DELAY_EMBEDDING'
WAVELETS = 'WAVELETS'

selected_models_10mins = {
    BOSS      : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(2000, 100, 100), learning_rate='adaptive', solver='adam')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_pyts_boss(word_size=4, n_bins=20, window_size=10, norm_mean=False,
                                               norm_std=False)
        ]
    },
    SIGNAL2VEC: {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(1000,), learning_rate='adaptive', solver='adam', activation='logistic')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=1)
        ]
    },
    PAA       : {
        CLF_MODELS        : [
            ExtraTreesClassifier(n_jobs=-1, n_estimators=500)
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_tslearn_paa(n_paa_segments=10, supports_approximation=True)
        ]
    },
    DFT       : {
        CLF_MODELS        : [
            ExtraTreesClassifier(n_jobs=-1, n_estimators=500)
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_pyts_dft(n_coefs=10, norm_mean=False, norm_std=False,
                                              supports_approximation=True)
        ]
    },
    SFA       : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(2000, 100, 100), learning_rate='adaptive', solver='adam')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_pyts_sfa(n_coefs=10, n_bins=5, norm_mean=False, norm_std=False)
        ]
    },
    SAX1D     : {
        CLF_MODELS        : [
            RandomForestClassifier(n_jobs=-1, n_estimators=100)
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_tslearn_one_d_sax(n_paa_segments=50, n_sax_symbols=10)
        ]
    },
    SAX       : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(2000), learning_rate='adaptive', solver='adam')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_tslearn_sax(n_paa_segments=50, n_sax_symbols=50, supports_approximation=True)
        ]
    }
}

selected_models_4h = {
    BOSS      : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(1000, 100), learning_rate='adaptive', solver='adam',
                          activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(1000, 100), learning_rate='adaptive', solver='adam',
                          activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(100,), learning_rate='adaptive', solver='adam', activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(100, 100), learning_rate='adaptive', solver='adam',
                          activation='logistic')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_pyts_boss(word_size=2, n_bins=26, window_size=10, norm_mean=False,
                                               norm_std=False),
            TransformerFactory.build_pyts_boss(word_size=2, n_bins=25, window_size=10, norm_mean=False,
                                               norm_std=False),
            TransformerFactory.build_pyts_boss(word_size=2, n_bins=26, window_size=10, norm_mean=False,
                                               norm_std=False),
            TransformerFactory.build_pyts_boss(word_size=2, n_bins=26, window_size=10, norm_mean=False,
                                               norm_std=False)
        ]
    },
    SIGNAL2VEC: {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(1000, 100), learning_rate='adaptive', solver='adam',
                          activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(1000,), learning_rate='adaptive', solver='adam', activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(1000,), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(100, 100, 100), learning_rate='adaptive', solver='adam'),
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=5),
            TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=10),
            TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=10),
            TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=10)
        ]
    },
    WEASEL    : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(100,), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(1000,), learning_rate='adaptive', solver='adam', activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(100,), learning_rate='adaptive', solver='adam', activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(100, 100, 100), learning_rate='adaptive', solver='adam')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_pyts_weasel(word_size=2, n_bins=4, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_weasel(word_size=2, n_bins=4, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_weasel(word_size=2, n_bins=4, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_weasel(word_size=2, n_bins=4, norm_mean=False, norm_std=False)
        ]
    },
    PAA       : {
        CLF_MODELS        : [
            ExtraTreesClassifier(n_jobs=-1, n_estimators=200),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=1000),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=2000),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=500)
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_tslearn_paa(n_paa_segments=10, supports_approximation=True),
            TransformerFactory.build_tslearn_paa(n_paa_segments=10, supports_approximation=True),
            TransformerFactory.build_tslearn_paa(n_paa_segments=10, supports_approximation=True),
            TransformerFactory.build_tslearn_paa(n_paa_segments=10, supports_approximation=True)
        ]
    },
    DFT       : {
        CLF_MODELS        : [
            ExtraTreesClassifier(n_jobs=-1, n_estimators=200),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=1000),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=2000),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=500)
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_pyts_dft(n_coefs=10, norm_mean=False, norm_std=False,
                                              supports_approximation=True),
            TransformerFactory.build_pyts_dft(n_coefs=10, norm_mean=False, norm_std=False,
                                              supports_approximation=True),
            TransformerFactory.build_pyts_dft(n_coefs=10, norm_mean=False, norm_std=False,
                                              supports_approximation=True),
            TransformerFactory.build_pyts_dft(n_coefs=10, norm_mean=False, norm_std=False,
                                              supports_approximation=True)
        ]
    },
    SFA       : {
        CLF_MODELS        : [
            ExtraTreesClassifier(n_jobs=-1, n_estimators=500),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=2000),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=1000),
            MLPClassifier(hidden_layer_sizes=(1000, 100), learning_rate='adaptive', solver='adam'),
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_pyts_sfa(n_coefs=10, n_bins=9, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_sfa(n_coefs=10, n_bins=9, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_sfa(n_coefs=10, n_bins=9, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_sfa(n_coefs=10, n_bins=9, norm_mean=False, norm_std=False)
        ]
    },
    SAX1D     : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(1000,), learning_rate='adaptive', solver='adam'),
            ExtraTreeClassifier(),
            ExtraTreeClassifier(),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=100)
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_tslearn_one_d_sax(n_paa_segments=50, n_sax_symbols=20),
            TransformerFactory.build_tslearn_one_d_sax(n_paa_segments=50, n_sax_symbols=20),
            TransformerFactory.build_tslearn_one_d_sax(n_paa_segments=50, n_sax_symbols=10),
            TransformerFactory.build_tslearn_one_d_sax(n_paa_segments=50, n_sax_symbols=50)
        ]
    },
    SAX       : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(1000,), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(100,), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(100,), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(1000, 100), learning_rate='adaptive', solver='adam')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_tslearn_sax(n_paa_segments=50, n_sax_symbols=50, supports_approximation=True),
            TransformerFactory.build_tslearn_sax(n_paa_segments=50, n_sax_symbols=10, supports_approximation=True),
            TransformerFactory.build_tslearn_sax(n_paa_segments=20, n_sax_symbols=50, supports_approximation=True),
            TransformerFactory.build_tslearn_sax(n_paa_segments=20, n_sax_symbols=10, supports_approximation=True)
        ]
    }
}

selected_models_8h = {
    BOSS      : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(2000, 100), learning_rate='adaptive', solver='adam',
                          activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(1000,), learning_rate='adaptive', solver='adam', activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(2000, 100), learning_rate='adaptive', solver='adam',
                          activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(1000, 100), learning_rate='adaptive', solver='adam',
                          activation='logistic')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_pyts_boss(word_size=2, n_bins=20, window_size=10, norm_mean=False,
                                               norm_std=False),
            TransformerFactory.build_pyts_boss(word_size=2, n_bins=20, window_size=10, norm_mean=False,
                                               norm_std=False),
            TransformerFactory.build_pyts_boss(word_size=4, n_bins=10, window_size=10, norm_mean=False,
                                               norm_std=False),
            TransformerFactory.build_pyts_boss(word_size=4, n_bins=10, window_size=10, norm_mean=False,
                                               norm_std=False)
        ]
    },
    SIGNAL2VEC: {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(2000, 100), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(100, 100, 100), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(1000, 100), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(1000,), learning_rate='adaptive', solver='adam', activation='logistic')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=50),
            TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=4),
            TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=1),
            TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=5)
        ]
    },
    WEASEL    : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(2000), learning_rate='adaptive', solver='adam', activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(100, 100, 100), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(2000, 100, 100), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(100, 100), learning_rate='adaptive', solver='adam')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_pyts_weasel(word_size=2, n_bins=4, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_weasel(word_size=2, n_bins=4, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_weasel(word_size=2, n_bins=4, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_weasel(word_size=2, n_bins=4, norm_mean=False, norm_std=False)
        ]
    },
    PAA       : {
        CLF_MODELS        : [
            ExtraTreesClassifier(n_jobs=-1, n_estimators=1000),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=500),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=2000),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=200)
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_tslearn_paa(n_paa_segments=10, supports_approximation=True),
            TransformerFactory.build_tslearn_paa(n_paa_segments=10, supports_approximation=True),
            TransformerFactory.build_tslearn_paa(n_paa_segments=10, supports_approximation=True),
            TransformerFactory.build_tslearn_paa(n_paa_segments=10, supports_approximation=True)
        ]
    },
    DFT       : {
        CLF_MODELS        : [
            ExtraTreesClassifier(n_jobs=-1, n_estimators=500),
            RandomForestClassifier(n_jobs=-1, n_estimators=100),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=100),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=2000)
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_pyts_dft(n_coefs=10, norm_mean=False, norm_std=False,
                                              supports_approximation=True),
            TransformerFactory.build_pyts_dft(n_coefs=10, norm_mean=False, norm_std=False,
                                              supports_approximation=True),
            TransformerFactory.build_pyts_dft(n_coefs=10, norm_mean=False, norm_std=False,
                                              supports_approximation=True),
            TransformerFactory.build_pyts_dft(n_coefs=10, norm_mean=False, norm_std=False,
                                              supports_approximation=True)
        ]
    },
    SFA       : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(100, 50, 100, 50), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(100, 100, 100), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(1000,), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(2000), learning_rate='adaptive', solver='adam')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_pyts_sfa(n_coefs=10, n_bins=5, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_sfa(n_coefs=10, n_bins=5, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_sfa(n_coefs=10, n_bins=5, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_sfa(n_coefs=10, n_bins=5, norm_mean=False, norm_std=False)
        ]
    },
    SAX1D     : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(100, 50, 100, 50), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(100, 100), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(2000), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(1000,), learning_rate='adaptive', solver='adam')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_tslearn_one_d_sax(n_paa_segments=50, n_sax_symbols=10),
            TransformerFactory.build_tslearn_one_d_sax(n_paa_segments=50, n_sax_symbols=10),
            TransformerFactory.build_tslearn_one_d_sax(n_paa_segments=50, n_sax_symbols=10),
            TransformerFactory.build_tslearn_one_d_sax(n_paa_segments=50, n_sax_symbols=20)
        ]
    },
    SAX       : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(2000, 100), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(2000), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(2000, 100), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(2000), learning_rate='adaptive', solver='adam')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_tslearn_sax(n_paa_segments=50, n_sax_symbols=100, supports_approximation=False),
            TransformerFactory.build_tslearn_sax(n_paa_segments=50, n_sax_symbols=10, supports_approximation=False),
            TransformerFactory.build_tslearn_sax(n_paa_segments=10, n_sax_symbols=20, supports_approximation=False),
            TransformerFactory.build_tslearn_sax(n_paa_segments=50, n_sax_symbols=20, supports_approximation=True)
        ]
    }
}

selected_models_1h = {
    BOSS      : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(100, 100, 100), learning_rate='adaptive', solver='adam',
                          activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(2000, 100, 100), learning_rate='adaptive', solver='adam',
                          activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(2000, 100), learning_rate='adaptive', solver='adam',
                          activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(1000, 100), learning_rate='adaptive', solver='adam',
                          activation='logistic')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_pyts_boss(word_size=4, n_bins=20, window_size=10, norm_mean=False,
                                               norm_std=False),
            TransformerFactory.build_pyts_boss(word_size=2, n_bins=20, window_size=10, norm_mean=False,
                                               norm_std=False),
            TransformerFactory.build_pyts_boss(word_size=2, n_bins=20, window_size=10, norm_mean=False,
                                               norm_std=False),
            TransformerFactory.build_pyts_boss(word_size=2, n_bins=20, window_size=10, norm_mean=False,
                                               norm_std=False)
        ]
    },
    SIGNAL2VEC: {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(1000, 100), learning_rate='adaptive', solver='adam',
                          activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(1000, 100), learning_rate='adaptive', solver='adam',
                          activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(2000, 100), learning_rate='adaptive', solver='adam',
                          activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(1000, 100), learning_rate='adaptive', solver='adam',
                          activation='logistic')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=2),
            TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=4),
            TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=4),
            TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=1)
        ]
    },
    WEASEL    : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(1000,), learning_rate='adaptive', solver='adam', activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(2000), learning_rate='adaptive', solver='adam', activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(100, 50, 100, 50), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(100,), learning_rate='adaptive', solver='adam', activation='logistic')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_pyts_weasel(word_size=2, n_bins=4, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_weasel(word_size=2, n_bins=4, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_weasel(word_size=2, n_bins=4, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_weasel(word_size=2, n_bins=4, norm_mean=False, norm_std=False)
        ]
    },
    PAA       : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(100, 100, 100), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(2000, 100, 100), learning_rate='adaptive', solver='adam'),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=100),
            MLPClassifier(hidden_layer_sizes=(100, 50, 100, 50), learning_rate='adaptive', solver='adam')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_tslearn_paa(n_paa_segments=10, supports_approximation=True),
            TransformerFactory.build_tslearn_paa(n_paa_segments=10, supports_approximation=True),
            TransformerFactory.build_tslearn_paa(n_paa_segments=10, supports_approximation=True),
            TransformerFactory.build_tslearn_paa(n_paa_segments=10, supports_approximation=True)
        ]
    },
    DFT       : {
        CLF_MODELS        : [
            ExtraTreesClassifier(n_jobs=-1, n_estimators=100),
            RandomForestClassifier(n_jobs=-1, n_estimators=100),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=500),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=1000)
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_pyts_dft(n_coefs=10, norm_mean=False, norm_std=False,
                                              supports_approximation=True),
            TransformerFactory.build_pyts_dft(n_coefs=10, norm_mean=False, norm_std=False,
                                              supports_approximation=True),
            TransformerFactory.build_pyts_dft(n_coefs=10, norm_mean=False, norm_std=False,
                                              supports_approximation=True),
            TransformerFactory.build_pyts_dft(n_coefs=10, norm_mean=False, norm_std=False,
                                              supports_approximation=True),
        ]
    },
    SFA       : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(2000, 100, 100), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(1000, 2000, 100), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(2000, 100), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(1000, 100), learning_rate='adaptive', solver='adam')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_pyts_sfa(n_coefs=10, n_bins=5, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_sfa(n_coefs=10, n_bins=5, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_sfa(n_coefs=10, n_bins=5, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_sfa(n_coefs=10, n_bins=5, norm_mean=False, norm_std=False)
        ]
    },
    SAX1D     : {
        CLF_MODELS        : [
            ExtraTreesClassifier(n_jobs=-1, n_estimators=100),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=200),
            RandomForestClassifier(n_jobs=-1, n_estimators=100),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=200)
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_tslearn_one_d_sax(n_paa_segments=50, n_sax_symbols=50),
            TransformerFactory.build_tslearn_one_d_sax(n_paa_segments=50, n_sax_symbols=10),
            TransformerFactory.build_tslearn_one_d_sax(n_paa_segments=50, n_sax_symbols=10),
            TransformerFactory.build_tslearn_one_d_sax(n_paa_segments=50, n_sax_symbols=50)
        ]
    },
    SAX       : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(100,), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(2000), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(100, 100, 100), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(1000,), learning_rate='adaptive', solver='adam'),
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_tslearn_sax(n_paa_segments=20, n_sax_symbols=10, supports_approximation=True),
            TransformerFactory.build_tslearn_sax(n_paa_segments=20, n_sax_symbols=50, supports_approximation=True),
            TransformerFactory.build_tslearn_sax(n_paa_segments=50, n_sax_symbols=10, supports_approximation=True),
            TransformerFactory.build_tslearn_sax(n_paa_segments=50, n_sax_symbols=50, supports_approximation=True)
        ]
    }
}

selected_models_2h = {
    BOSS      : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(2000, 100), learning_rate='adaptive', solver='adam',
                          activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(1000, 100), learning_rate='adaptive', solver='adam',
                          activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(1000, 2000, 100), learning_rate='adaptive', solver='adam',
                          activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(100,), learning_rate='adaptive', solver='adam', activation='logistic')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_pyts_boss(word_size=2, n_bins=20, window_size=10, norm_mean=False,
                                               norm_std=False),
            TransformerFactory.build_pyts_boss(word_size=2, n_bins=20, window_size=10, norm_mean=False,
                                               norm_std=False),
            TransformerFactory.build_pyts_boss(word_size=2, n_bins=20, window_size=10, norm_mean=False,
                                               norm_std=False),
            TransformerFactory.build_pyts_boss(word_size=2, n_bins=20, window_size=10, norm_mean=False,
                                               norm_std=False)
        ]
    },
    SIGNAL2VEC: {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(2000, 100), learning_rate='adaptive', solver='adam',
                          activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(2000), learning_rate='adaptive', solver='adam', activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(1000, 100), learning_rate='adaptive', solver='adam',
                          activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(2000, 100), learning_rate='adaptive', solver='adam',
                          activation='logistic')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=4),
            TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=4),
            TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=4),
            TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=5)
        ]
    },
    WEASEL    : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(2000), learning_rate='adaptive', solver='adam', activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(1000,), learning_rate='adaptive', solver='adam', activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(100, 50, 100, 50), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(100,), learning_rate='adaptive', solver='adam', activation='logistic')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_pyts_weasel(word_size=2, n_bins=4, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_weasel(word_size=2, n_bins=4, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_weasel(word_size=2, n_bins=4, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_weasel(word_size=2, n_bins=4, norm_mean=False, norm_std=False)
        ]
    },
    PAA       : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(100, 100, 100), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(2000, 100, 100), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(100, 50, 100, 50), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(2000), learning_rate='adaptive', solver='adam', activation='logistic')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_tslearn_paa(n_paa_segments=10, supports_approximation=True),
            TransformerFactory.build_tslearn_paa(n_paa_segments=10, supports_approximation=True),
            TransformerFactory.build_tslearn_paa(n_paa_segments=10, supports_approximation=True),
            TransformerFactory.build_tslearn_paa(n_paa_segments=10, supports_approximation=True)
        ]
    },
    DFT       : {
        CLF_MODELS        : [

        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_pyts_dft(n_coefs=10, norm_mean=False, norm_std=False,
                                              supports_approximation=True),
            TransformerFactory.build_pyts_dft(n_coefs=10, norm_mean=False, norm_std=False,
                                              supports_approximation=True),
            TransformerFactory.build_pyts_dft(n_coefs=10, norm_mean=False, norm_std=False,
                                              supports_approximation=True),
            TransformerFactory.build_pyts_dft(n_coefs=10, norm_mean=False, norm_std=False,
                                              supports_approximation=True)
        ]
    },
    SFA       : {
        CLF_MODELS        : [
            ExtraTreesClassifier(n_jobs=-1, n_estimators=500),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=2000),
            RandomForestClassifier(n_jobs=-1, n_estimators=100),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=1000)
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_pyts_sfa(n_coefs=10, n_bins=5, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_sfa(n_coefs=10, n_bins=5, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_sfa(n_coefs=10, n_bins=5, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_sfa(n_coefs=10, n_bins=5, norm_mean=False, norm_std=False)
        ]
    },
    SAX1D     : {
        CLF_MODELS        : [
            ExtraTreesClassifier(n_jobs=-1, n_estimators=1000),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=2000),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=2000),
            RandomForestClassifier(n_jobs=-1, n_estimators=100)
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_tslearn_one_d_sax(n_paa_segments=50, n_sax_symbols=10),
            TransformerFactory.build_tslearn_one_d_sax(n_paa_segments=50, n_sax_symbols=10),
            TransformerFactory.build_tslearn_one_d_sax(n_paa_segments=50, n_sax_symbols=20),
            TransformerFactory.build_tslearn_one_d_sax(n_paa_segments=50, n_sax_symbols=20)
        ]
    },
    SAX       : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(2000), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(1000, 100), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(1000, 100), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(2000, 100), learning_rate='adaptive', solver='adam')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_tslearn_sax(n_paa_segments=50, n_sax_symbols=50, supports_approximation=True),
            TransformerFactory.build_tslearn_sax(n_paa_segments=50, n_sax_symbols=10, supports_approximation=True),
            TransformerFactory.build_tslearn_sax(n_paa_segments=50, n_sax_symbols=50, supports_approximation=True),
            TransformerFactory.build_tslearn_sax(n_paa_segments=50, n_sax_symbols=50, supports_approximation=True)
        ]
    }
}

selected_models_24h = {
    BOSS      : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(2000), learning_rate='adaptive', solver='adam', activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(2000, 100), learning_rate='adaptive', solver='adam',
                          activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(2000), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(2000), learning_rate='adaptive', solver='adam')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_pyts_boss(word_size=4, n_bins=5, window_size=10, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_boss(word_size=4, n_bins=5, window_size=10, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_boss(word_size=4, n_bins=10, window_size=10, norm_mean=False,
                                               norm_std=False),
            TransformerFactory.build_pyts_boss(word_size=4, n_bins=5, window_size=10, norm_mean=False, norm_std=False)
        ]
    },
    SIGNAL2VEC: {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(1000, 100), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(2000), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(1000, 100), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(2000, 100, 100), learning_rate='adaptive', solver='adam')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=4),
            TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=4),
            TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=5),
            TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=2)
        ]
    },
    WEASEL    : {
        CLF_MODELS        : [
        ],
        TRANSFORMER_MODELS: [
        ]
    },
    PAA       : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(1000, 100), learning_rate='adaptive', solver='adam',
                          activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(2000), learning_rate='adaptive', solver='adam', activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(1000,), learning_rate='adaptive', solver='adam', activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(2000, 100), learning_rate='adaptive', solver='adam',
                          activation='logistic')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_tslearn_paa(n_paa_segments=10, supports_approximation=True),
            TransformerFactory.build_tslearn_paa(n_paa_segments=10, supports_approximation=True),
            TransformerFactory.build_tslearn_paa(n_paa_segments=10, supports_approximation=True),
            TransformerFactory.build_tslearn_paa(n_paa_segments=10, supports_approximation=True)
        ]
    },
    DFT       : {
        CLF_MODELS        : [
            ExtraTreesClassifier(n_jobs=-1, n_estimators=100),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=2000),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=500),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=200)
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_pyts_dft(n_coefs=10, norm_mean=False, norm_std=False,
                                              supports_approximation=True),
            TransformerFactory.build_pyts_dft(n_coefs=10, norm_mean=False, norm_std=False,
                                              supports_approximation=True),
            TransformerFactory.build_pyts_dft(n_coefs=10, norm_mean=False, norm_std=False,
                                              supports_approximation=True),
            TransformerFactory.build_pyts_dft(n_coefs=10, norm_mean=False, norm_std=False,
                                              supports_approximation=True)
        ]
    },
    SFA       : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(100,), learning_rate='adaptive', solver='adam'),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=1000),
            RandomForestClassifier(n_jobs=-1, n_estimators=200),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=100)
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_pyts_sfa(n_coefs=10, n_bins=5, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_sfa(n_coefs=10, n_bins=5, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_sfa(n_coefs=10, n_bins=5, norm_mean=False, norm_std=False),
            TransformerFactory.build_pyts_sfa(n_coefs=10, n_bins=5, norm_mean=False, norm_std=False)
        ]
    },
    SAX1D     : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(1000,), learning_rate='adaptive', solver='adam', activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(2000), learning_rate='adaptive', solver='adam', activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(2000), learning_rate='adaptive', solver='adam', activation='logistic'),
            MLPClassifier(hidden_layer_sizes=(1000,), learning_rate='adaptive', solver='adam', activation='logistic')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_tslearn_one_d_sax(n_paa_segments=50, n_sax_symbols=10),
            TransformerFactory.build_tslearn_one_d_sax(n_paa_segments=50, n_sax_symbols=20),
            TransformerFactory.build_tslearn_one_d_sax(n_paa_segments=50, n_sax_symbols=50),
            TransformerFactory.build_tslearn_one_d_sax(n_paa_segments=50, n_sax_symbols=100)
        ]
    },
    SAX       : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(100, 100, 100), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(2000, 100), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(100, 100), learning_rate='adaptive', solver='adam'),
            MLPClassifier(hidden_layer_sizes=(2000), learning_rate='adaptive', solver='adam')
        ],
        TRANSFORMER_MODELS: [
            TransformerFactory.build_tslearn_sax(n_paa_segments=50, n_sax_symbols=10, supports_approximation=True),
            TransformerFactory.build_tslearn_sax(n_paa_segments=20, n_sax_symbols=50, supports_approximation=True),
            TransformerFactory.build_tslearn_sax(n_paa_segments=20, n_sax_symbols=10, supports_approximation=True),
            TransformerFactory.build_tslearn_sax(n_paa_segments=20, n_sax_symbols=50, supports_approximation=True)
        ]
    }
}

model_selection_clf_list = [
    MLPClassifier(hidden_layer_sizes=(1000,), learning_rate='adaptive', solver='adam',
                  activation='logistic'),
    MLPClassifier(hidden_layer_sizes=(1000, 100), learning_rate='adaptive', solver='adam',
                  activation='logistic'),
    MLPClassifier(hidden_layer_sizes=(2000), learning_rate='adaptive', solver='adam',
                  activation='logistic')
]

model_selection_transformers = [
    TransformerFactory.build_pyts_boss(word_size=2, n_bins=5, window_size=10, norm_mean=False, norm_std=False),
    TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=2),
    TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=1),
    TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, transformer_type=TransformerType.transform)
]

model_selection_mlknn = [MLkNN(k=1, s=1.0, ignore_first_neighbours=0),
                         MLkNN(k=3, s=1.0, ignore_first_neighbours=0),
                         MLkNN(k=10, s=1.0, ignore_first_neighbours=0),
                         MLkNN(k=20, s=1.0, ignore_first_neighbours=0),

                         MLkNN(k=1, s=0.5, ignore_first_neighbours=0),
                         MLkNN(k=3, s=0.5, ignore_first_neighbours=0),
                         MLkNN(k=10, s=0.5, ignore_first_neighbours=0),
                         MLkNN(k=20, s=0.5, ignore_first_neighbours=0),

                         MLkNN(k=1, s=0.7, ignore_first_neighbours=0),
                         MLkNN(k=3, s=0.7, ignore_first_neighbours=0),
                         MLkNN(k=10, s=0.7, ignore_first_neighbours=0),
                         MLkNN(k=20, s=0.7, ignore_first_neighbours=0)
                         ]

model_selection_rakel = [
    RakelD(MLPClassifier(hidden_layer_sizes=(100, 100, 100), learning_rate='adaptive', solver='adam')),
    RakelD(MLPClassifier(hidden_layer_sizes=(100, 100, 100), learning_rate='adaptive', solver='adam'), labelset_size=5),
    RakelD(MLPClassifier(hidden_layer_sizes=(100, 100), learning_rate='adaptive', solver='adam')),
    RakelD(MLPClassifier(hidden_layer_sizes=(100, 100), learning_rate='adaptive', solver='adam'), labelset_size=5),
    RakelD(MLPClassifier(hidden_layer_sizes=(2000, 100), learning_rate='adaptive', solver='adam')),
    RakelD(MLPClassifier(hidden_layer_sizes=(2000, 100), learning_rate='adaptive', solver='adam'), labelset_size=5),
    RakelD(MLPClassifier(hidden_layer_sizes=(2000), learning_rate='adaptive', solver='adam')),
    RakelD(MLPClassifier(hidden_layer_sizes=(2000), learning_rate='adaptive', solver='adam'), labelset_size=5),
    RakelD(base_classifier=GaussianNB(), base_classifier_require_dense=[True, True], labelset_size=3),
    RakelD(base_classifier=GaussianNB(), base_classifier_require_dense=[True, True], labelset_size=5),
    RakelD(base_classifier=GaussianNB(), base_classifier_require_dense=[True, True], labelset_size=7)
]

model_selection_wavelets = [
    TransformerFactory.build_wavelet(),
    TransformerFactory.build_wavelet(drop_cA=True)
]

model_selection_delay_embeddings = [
    TransformerFactory.build_delay_embedding(delay_in_seconds=30, dimension=6),
    TransformerFactory.build_delay_embedding(delay_in_seconds=32, dimension=8),
    TransformerFactory.build_delay_embedding(delay_in_seconds=6, dimension=8),
    TransformerFactory.build_delay_embedding(delay_in_seconds=12, dimension=8)
]

cv_signal2vec = [TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=1)]
cv_signal2vec_clf = [MLPClassifier(hidden_layer_sizes=(1000,), learning_rate='adaptive',
                                   solver='adam', activation='logistic')]
cv_boss_clf = [MLPClassifier(hidden_layer_sizes=(2000, 100, 100), learning_rate='adaptive', solver='adam')]
cv_boss = [TransformerFactory.build_pyts_boss(word_size=2, n_bins=2, window_size=10,
                                              norm_mean=False, norm_std=False)]

state_of_the_art = {
    SIGNAL2VEC          : {
        CLF_MODELS        : [MLPClassifier(hidden_layer_sizes=(1000,), learning_rate='adaptive',
                                           solver='adam', activation='logistic')],
        TRANSFORMER_MODELS: [TransformerFactory.build_signal2vec(SAVED_MODEL, PATH_SIGNAL2VEC, num_of_vectors=1)]
    },
    WAVELETS            : {
        CLF_MODELS        : [MLkNN(ignore_first_neighbours=0, k=3, s=1.0),
                             RakelD(MLPClassifier(hidden_layer_sizes=(100, 100, 100), learning_rate='adaptive',
                                                  solver='adam'), labelset_size=5)],
        TRANSFORMER_MODELS: [TransformerFactory.build_wavelet(), TransformerFactory.build_wavelet()]
    },
    TIME_DELAY_EMBEDDING: {
        CLF_MODELS        : [
            MLkNN(ignore_first_neighbours=0, k=3, s=1.0),
            RakelD(MLPClassifier(hidden_layer_sizes=(100, 100, 100), learning_rate='adaptive',
                                 solver='adam'), labelset_size=5)
        ],
        TRANSFORMER_MODELS: [TransformerFactory.build_delay_embedding(delay_in_seconds=30, dimension=6),
                             TransformerFactory.build_delay_embedding(delay_in_seconds=30, dimension=6)
                             ]
    },
    BOSS                : {
        CLF_MODELS        : [
            MLPClassifier(hidden_layer_sizes=(2000, 100, 100), learning_rate='adaptive', solver='adam')],
        TRANSFORMER_MODELS: [TransformerFactory.build_pyts_boss(word_size=2, n_bins=4, window_size=10,
                                                                norm_mean=False, norm_std=False)]
    }
}
