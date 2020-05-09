# MULTI-NILM: A novel framework for multi-label Non-Intrusive Load Monitoring

## Description
This repository is based  on our paper with title: 
["On time series representations for multi-label NILM"](https://rdcu.be/b3Vh2) [1] 
and it can be used to replicate the experiments. It defines a framework for multi-label NILM systems and includes the following time series
representations: **Signal2Vec, BOSS, SFA, WEASEL, DFT, SAX, 1d-SAX, PAA**; 
and an implementation of **delay embedding** using Taken's theorem. Feel free to reuse, modify and extend this repository.

## Multi-NILM framework
Multi-nilm is a novel framework for efficient non-intrusive load monitoring systems. 
It has three inherent properties:
- It utilizes a data representation for sufficient dimensionality reduction.
- It uses lightweight disaggregation models.
- It tackles the disaggregation problem as a multi-label classification problem.

## Examples
Examples of experiments can be found under the directory _experiments_. 
The module [experiments.py](experiments/experiments.py) defines three types of experiments (_GenericExperiment, ModelSelectionExperiment_ 
and _REDDModelSelectionExperiment_). You can also create your own 
experiment by extending the abstract class _nilmlab.lab.Experiment_.

After defining an experiment it requires only a few lines of code to setup and configure it. 
All files with names _run*.py_ are specific implementations that can be used as a reference.
In order to run any of them it is as simple as: 
```python
python -m experiments.run_generic_experiment
```
The results are saved under the directory _results_ as a csv file containing information about the 
setup, the source of the data, the parameters, the classification models, the performance and others.

## Data

Currently only **REDD** and **UK DALE** are supported, which have to be downloaded manually. 
The popular **NILMTK** toolkit is used for reading the energy data.

## Project structure
A detailed structure of the project is presented below. The key points are:
   - 📂 __data\_exploration__: Contains helpful notebooks e.g. how to define delay embedding parameters.
   - 📂 __datasources__: Includes modules related to data e.g. loading using nilmtk, processing labels and others. 
   - 📂 __experiments__: Defines some experiments such as model selection and has examples on how to run the 
   defined experiments. 
   - 📂 __nilmlab__: This is the main code which encapsulates all the logic of the proposed framework 
   and implements various time series representations.
   - 📂 __pretrained\_models__: Any pretrained models that are used for Signal2Vec [1,2].
   - 📂 __results__: Results of the experiments will be saved in this directory.
   - 📂 __utils__: Various tools that have been developed to support the implementation of the various algorithms.


- 📂 __multi\-nilm__
   - 📄 [LICENSE](LICENSE)
   - 📄 [README.md](README.md)
   - 📄 [createtree.sh](createtree.sh)
   - 📂 __data\_exploration__: 
     - 📄 [\_\_init\_\_.py](data_exploration/__init__.py)
     - 📂 __time\_delay\_embedding__
       - 📄 [delay\_embedding\_parameterization\-redd.ipynb](data_exploration/time_delay_embedding/delay_embedding_parameterization-redd.ipynb)
       - 📄 [delay\_embedding\_parameterization\-uk\_dale.ipynb](data_exploration/time_delay_embedding/delay_embedding_parameterization-uk_dale.ipynb)
   - 📂 __datasources__
     - 📄 [\_\_init\_\_.py](datasources/__init__.py)
     - 📄 [datasource.py](datasources/datasource.py)
     - 📄 [labels\_factory.py](datasources/labels_factory.py)
     - 📄 [paths\_manager.py](datasources/paths_manager.py)
   - 📂 __experiments__
     - 📄 [\_\_init\_\_.py](experiments/__init__.py)
     - 📄 [experiments.py](experiments/experiments.py)
     - 📄 [run\_generic\_experiment.py](experiments/run_generic_experiment.py)
     - 📄 [run\_model\_selection\_1h\_cv5.py](experiments/run_model_selection_1h_cv5.py)
     - 📄 [run\_model\_selection\_5mins\_cv5.py](experiments/run_model_selection_5mins_cv5.py)
     - 📄 [run\_model\_selection\_5mins\_cv5\_redd.py](experiments/run_model_selection_5mins_cv5_redd.py)
     - 📄 [run\_state\_of\_the\_art.py](experiments/run_state_of_the_art.py)
   - 📂 __nilmlab__
     - 📄 [\_\_init\_\_.py](nilmlab/__init__.py)
     - 📄 [exp\_model\_list.py](nilmlab/exp_model_list.py)
     - 📄 [factories.py](nilmlab/factories.py)
     - 📄 [lab.py](nilmlab/lab.py)
     - 📄 [lab\_exceptions.py](nilmlab/lab_exceptions.py)
     - 📄 [tstransformers.py](nilmlab/tstransformers.py)
   - 📂 __pretrained\_models__
     - 📄 [clf\-v1.pkl](pretrained_models/clf-v1.pkl)
     - 📄 [signal2vec\-v1.csv](pretrained_models/signal2vec-v1.csv)
   - 📄 [requirements.txt](requirements.txt)
   - 📂 __results__
     - 📄 [\_\_init\_\_.py](results/__init__.py)
   - 📄 [tree.md](tree.md)
   - 📂 __utils__
     - 📄 [\_\_init\_\_.py](utils/__init__.py)
     - 📄 [chaotic\_toolkit.py](utils/chaotic_toolkit.py)
     - 📄 [logger.py](utils/logger.py)
     - 📄 [multiprocessing\_tools.py](utils/multiprocessing_tools.py)

## Dependencies

The code has been developed using python3.6 and the dependencies can be found in [requirements.txt](requirements.txt).
- numpy~=1.18.1
- scikit-learn~=0.21.3
- pandas~=1.0.1
- loguru~=0.4.1
- nilmtk~=0.4.0
- pyts~=0.10.0 https://github.com/johannfaouzi/pyts
- tslearn~=0.3.0 https://github.com/tslearn-team/tslearn
- scikit-multilearn~=0.2.0 http://scikit.ml/
- psutil~=5.6.7
- matplotlib~=3.2.0
- fuzzywuzzy~=0.17.0
- numba~=0.48.0
- PyWavelets https://pywavelets.readthedocs.io/en/latest/install.html

## Licence

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


## References
1. Nalmpantis, C., Vrakas, D. On time series representations for multi-label NILM. Neural Comput & Applic (2020). https://doi.org/10.1007/s00521-020-04916-5
2. Nalmpantis, C., & Vrakas, D. (2019, May). Signal2Vec: Time Series Embedding Representation. In International Conference on Engineering Applications of Neural Networks (pp. 80-90). Springer, Cham. https://doi.org/10.1007/978-3-030-20257-6_7
