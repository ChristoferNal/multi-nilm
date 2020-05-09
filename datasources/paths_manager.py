import os

dirname = os.path.dirname(__file__)
UK_DALE = os.path.join(dirname, '../../Datasets/UKDALE/ukdale.h5')
REDD = os.path.join(dirname, '../../Datasets/REDD/redd.h5')

SAVED_MODEL = os.path.join(dirname, "../pretrained_models/clf-v1.pkl")
PATH_SIGNAL2VEC = os.path.join(dirname, '../pretrained_models/signal2vec-v1.csv')
