# Acoustic Heterogeneity from Soundscapes using Graph-based Neural Networks for Ecological Monitoring

Codes from my doctoral thesis "Acoustic Heterogeneity from Soundscapes using Graph-based Neural Networks for Ecological Monitoring."

The simplest method to use the codes is to initially establish a local environment using the `conda`:
```shell
conda create --name phd-thesis python=3.10.12
```
The, within the environment, you will need to run the following command in the Python repository to install all the necessary dependencies.
```shell
pip install -r requirements.txt
```

## Repository structure

- `data/`
  - Contains coverage maps of the study site, and information on coverage proportion, longitude, and latitude for each recorder.
- `Interpretability_Results/`
  - Interpretability results using the Eigen-CAM technique for all recorders
- `old_stuff/`
  - It contains some code used in preliminary experiments, where initial ideas were explored.
- `GraphDataset.py`, `models.py`, and `modelsCAM.py`: Dataset module for generating temporal blocks, model classes, and model classes with interpretability, respectively.
- `GVAE_semi_eval_post.ipynb`, `GVAE_train_multits.ipynb`, `GVAE_VGGish_CAMtests.ipynb`, and `GVAE_VGGish_tests.ipynb`: notebooks for training, evaluation, and interpretation of Graph VAE models.

*Remark*: Due to the size of the files, the data features and trained models are not hosted in this repository. If required, they will be shared upon reasonable request.
