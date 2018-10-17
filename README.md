# Adversarial generation of gene expression data

This work is submitted as part requirement for the MSc degree in Machine Learning at University College London

## Overview

The problem of reverse engineering gene regulatory networks from high-throughput expression data is one of the biggest challenges in bioinformatics. In order to benchmark network inference algorithms, simulators of well-characterized expression datasets are often required. However, existing simulators have been criticized because they fail to emulate key properties of gene expression data (Maier et al., 2013).

In this thesis we address two problems. First, we study and propose mechanisms to faithfully assess the realism of a synthetic expression dataset. Second, we design an adversarial simulator of expression data, gGAN, based on a generative adversarial network (Goodfellow et al., 2014). We show that our model outperforms existing simulators by a large margin in terms of the realism of the generated data. More importantly, our results show that gGAN is, to our best knowledge, the first simulator that passes the Turing test for gene expression data proposed by Maier et al. (2013).


## Prerequisites
Python 3.5

## Installation
1. Install TensorFlow (see https://www.tensorflow.org/install/)
2. Run `sudo pip install -r requirements.txt`

## Key files
- `src/ggan.py`: Trains the GAN, and stores some data.
- `src/sampling.py`: Samples data from a saved model.
- `src/validation.ipynb`: Evaluates synthetic dataset against the *E. coli* M3D train dataset.
- `src/ggan_analysis.ipynb`: Evaluates synthetic dataset against the *E. coli* M3D test dataset.
