#' Construct a mistnet model
#' 
#' This function creates a \code{network} object for fitting a mistnet model.
#' 
#' @param x a \code{numeric} \code{matrix} of predictor variables.  One row
#'  per example, one column per predictive feature.
#' @param y a \code{matrix} of responses to \code{x}.  One row per example, one
#'  column per response variable.
#' @param hidden.dims an \code{integer} \code{vector}. Each element specifies 
#'  the number of hidden units in a hidden \code{layer} of the resulting
#'  \code{network}. If your network should not have any hidden layers, 
#'  \code{hidden.dims} should be \code{NULL}.
#' @param n.ranef The number of latent random variables to include in the first
#'  layer of the \code{network}.
#' @param nonlinearity.names A character vector with the names of the 
#'  \code{nonlinearity} objects to use (one per \code{layer})  Currently 
#'  supported values include "sigmoid", "rectify", "exp", "linear", and "mf_mrf".
#'  See \code{\link{nonlinearity}}.
#' @param loss.name Currently supported values include "crossEntropy" (for 
#'  bernoulli likelihood), "binomialLoss", "poissonLoss", "squaredLoss", 
#'  and "mrfLoss".
#' param priors
#' 
#' @seealso \code{\link{network}}

mistnet = function(
  x,
  y,
  hidden.dims,
  n.ranef,
  nonlinearity.names,
  loss,
  priors = replicate(
    length(nonlinearity.names), 
    gaussianPrior(mean = 0, var = 1),
    simplify = FALSE
  ),
  updater.name = "sgd",
  updater.arguments = list(learning.rate = .001, momentum = .9),
  ranefSample = gaussianRanefSample,
  n.importance.samples = 25,
  minibatch.size = 20,
  training.iterations = 0
){
  n.layers = length(nonlinearity.names)
  assert_that(length(priors) == n.layers)
  assert_that((length(hidden.dims) + 1L) == n.layers)
  
  if (!is.null(hidden.dims)){
    hidden.dims = safe.as.integer(hidden.dims)
  }
  n.ranef = safe.as.integer(n.ranef)
  minibatch.size = safe.as.integer(minibatch.size)
  n.importance.samples = safe.as.integer(n.importance.samples)
  
  network.dims = c(ncol(x) + n.ranef, hidden.dims, ncol(y))
  
  stopifnot(nrow(x) == nrow(y))
  dataset.size = nrow(x)
  
  net = network$new( 
    x = x,
    y = y,
    layers = lapply(
      1:n.layers,
      function(i){
        createLayer(
          n.in = network.dims[[i]],
          n.out = network.dims[[i + 1]],
          prior = priors[[i]],
          nonlinearity.name = nonlinearity.names[[i]],
          updater.name = updater.name,
          updater.arguments = updater.arguments,
          minibatch.size = minibatch.size,
          n.importance.samples = n.importance.samples
        )
      }
    ),
    n.layers = n.layers,
    dataset.size = dataset.size,
    minibatch.size = minibatch.size,
    n.importance.samples = n.importance.samples,
    loss = loss$loss,
    lossGradient = loss$grad,
    ranefSample = ranefSample,
    n.ranef = n.ranef,
    completed.iterations = 0L
  ) 
  
  colnames(net$layers[[net$n.layers]]$coefficients) = colnames(y)
  dimnames(net$layers[[net$n.layers]]$outputs) = list(NULL, colnames(y), NULL)
  
  
  # Coefficients can't all start at zero! Perhaps sample coefficients from their
  # prior?
  # Final layer's biases shouldn't be zero either!
  
  net$fit(training.iterations)
  
  return(net)
}


createLayer = function(
  n.inputs,
  n.outputs,
  prior,
  nonlinearity.name,
  updater.name,
  updater.arguments,
  minibatch.size,
  n.importance.samples
){
  out = layer$new(
    coef.dim = c(n.inputs, n.outputs),
    coefficients = matrix(0, nrow = n.inputs, ncol = n.outputs),
    biases = matrix(rep(0, n.outputs), nrow = 1),
    coef.updater = do.call(
      new,
      c(
        list(
          Class = paste(updater.name, "updater", sep = "."),
          delta = matrix(0, nrow = n.inputs, ncol = n.outputs)
        ),
        updater.arguments
      )
    ),
    bias.updater = do.call(
      new,
      c(
        list(
          Class = paste(updater.name, "updater", sep = "."),
          delta = matrix(0, nrow = 1, ncol = n.outputs)
        ),
        updater.arguments
      )
    ),
    nonlinearity = new(
      paste(nonlinearity.name, "nonlinearity", sep = ".")
    ),
    prior = prior
  )
  out$resetState(minibatch.size, n.importance.samples)
  
  out
}

defineLayer = function(nonlinearity, size, prior){
  list(nonlinearity = nonlinearity, size = size, prior = prior)
}
