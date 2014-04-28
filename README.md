mistnet: Structured prediction with neural networks in R
=========

Mistnet is an R package that produces probability densities over multivariate outcomes.  Ecologists can use it to define probability densities over possible species assemblages.

**Note that this software is under active development and that there will be major changes to the user interface and default methods in the next few months.**  Development should stabilize some time this summer (2014).

Mistnet models are *stochastic* neural networks, meaning that they include stochastic latent variables (like random effects) that account for correlations among the outcome variables that cannot be explained by the inputs.

The model uses a Generalized Expectation Maximization approach to model fitting (maximized penalized likelihood), as described in [this paper](http://www-etud.iro.umontreal.ca/~goodfeli/sfnn_wk.pdf) by Tang and Salakhutdinov at ICML 2013.
