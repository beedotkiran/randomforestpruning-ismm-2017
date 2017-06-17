# Cost-Complexity pruning of Random Forests ISMM 2017
* [Paper](https://arxiv.org/abs/1703.05430), [Poster](https://beedotkiran.github.io/pruning_posterISMM2017.pdf), 
* Only Classification tasks have been evaluated.

## Datasets
* Download [winequality dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality), and other datasets and change utils.py to add new datasets to test out the pruning algorithm.

## Todo 
* Calculate the test leaves id at the same time as train leaves id
  * Then predict with optimal leaf labeling

## New references
* Impact of subsampling and pruning on random forests [Paper](https://arxiv.org/abs/1603.04261)
* Understanding variable importances in forests of randomized trees [Paper](https://papers.nips.cc/paper/4928-understanding-variable-importances-in-forests-of-randomized-trees.pdf)
