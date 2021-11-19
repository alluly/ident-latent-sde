# Identifying Latent Stochastic Differential Equations

This repo provides code for learning latent stochastic differential equations with variational autoencoders. The main idea is given a time series, uncover the latent differential equation that governs the time series. 

To get started, check out some of the examples in the notebook to get an idea of how the code works. 

For synthetic data, the code uses a `yaml` file that details the necessary components of the experiment to generate data and train the network. 

For real data, you must specify a subclass of the `SDEDataset` to define the aspects of your data. 

### Main functions

- To run the code with a particular `yaml` file, simply run `python3 train.py -f experiment.yaml`. 
- To compute some statistics regarding how well the learned drift recovered the SDE, run `python3 comp_crlb.py -f experiment.yaml`. 
- To test the autoencoder reconstruction capabilities on new data, run `python3 test.py -f experiment.yaml`. 
- To analyze all the results run `python3 analyze.py -f experiment.yaml`

These are the four main functions that are used to compute properties of an experiment. However, usually the most interesting one is simply `train.py` to see what kind of SDE may be generating your data. 
 
The code is written in `PyTorch` and uses mainly dependencies of `PyTorch`. 




