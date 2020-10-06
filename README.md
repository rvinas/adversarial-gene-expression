# Adversarial generation of gene expression data

This work is submitted as part requirement for the MSc degree in Machine Learning at University College London

## Overview

High-throughput gene expression can be used to address a wide range of fundamental biological problems, but datasets of an appropriate size are often unavailable. Moreover, existing transcriptomics simulators have been criticised because they fail to emulate key properties of gene expression data. In this paper, we develop a method based on a conditional generative adversarial network to generate realistic transcriptomics data for _E. coli_ and humans. We assess the performance of our approach across several tissues and cancer types. We show that our model preserves several gene expression properties significantly better than widely used simulators such as SynTReN or GeneNetWeaver. The synthetic data preserves tissue and cancer-specific properties of transcriptomics data. Moreover, it exhibits real gene clusters and ontologies both at local and global scales, suggesting that the model learns to approximate the gene expression manifold in a biologically meaningful way.


## Main figures


[UMAP_combined.pdf](figures/UMAP_combined.pdf)

[dist_dendr_combined_ellipses.pdf.pdf](figures/dist_dendr_combined_ellipses.pdf.pdf)
