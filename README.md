mistnet: Structured prediction with neural networks in R
=========

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.12423.png)](http://dx.doi.org/10.5281/zenodo.12423) 
[![Travis-CI Build Status](https://travis-ci.org/davharris/mistnet.svg?branch=master)](https://travis-ci.org/davharris/mistnet)

Mistnet is an R package that produces probability densities over multivariate outcomes.  Ecologists can use it to define probability densities over possible species assemblages, as described in [this paper](http://onlinelibrary.wiley.com/doi/10.1111/2041-210X.12332/full) I wrote for *Methods and Ecology and Evolution*.

Mistnet models are *stochastic* neural networks, meaning that they include stochastic latent variables (like random effects) that account for correlations among the outcome variables that cannot be explained by the inputs.

The model uses a Generalized Expectation Maximization approach to model fitting (maximized penalized likelihood), as described in [this paper](http://papers.nips.cc/paper/5026-learning-stochastic-feedforward-neural-networks.pdf) from Tang and Salakhutdinov at NIPS 2013 and in the *Methods* paper referred to above.
