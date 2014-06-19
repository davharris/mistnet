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
  layer.definitions,
  loss,
  updater = sgd.updater(learning.rate = .001, momentum = .9),
  sampler = gaussianSampler(ncol = 10, sd = 1),
  n.importance.samples = 25,
  n.minibatch = 20,
  training.iterations = 0
){
  n.minibatch = safe.as.integer(n.minibatch)
  n.importance.samples = safe.as.integer(n.importance.samples)
  n.layers = length(layer.definitions)
  
  
  network.dims = c(
    ncol(x) + with(environment(sampler), ncol), 
    sapply(layer.definitions, function(x) x$size)
  )
  
  if(network.dims[length(network.dims)] != ncol(y)){
    stop("The number of outputs in the last layer must equal the number of columns in y")
  }
  
  assert_that(nrow(x) == nrow(y))
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
          prior = layer.definitions[[i]]$prior,
          nonlinearity = layer.definitions[[i]]$nonlinearity,
          updater = updater,
          n.minibatch = n.minibatch,
          n.importance.samples = n.importance.samples
        )
      }
    ),
    n.layers = n.layers,
    dataset.size = dataset.size,
    n.minibatch = n.minibatch,
    n.importance.samples = n.importance.samples,
    loss = loss$loss,
    lossGradient = loss$grad,
    sampler = sampler,
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
  nonlinearity,
  updater,
  n.minibatch,
  n.importance.samples
){
  coef.updater = updater$copy()
  coef.updater$delta = matrix(0, nrow = n.inputs, ncol = n.outputs)
  
  bias.updater = updater$copy()
  bias.updater$delta = matrix(0, nrow = 1, ncol = n.outputs)
  
  out = layer$new(
    coef.dim = c(n.inputs, n.outputs),
    coefficients = matrix(0, nrow = n.inputs, ncol = n.outputs),
    biases = matrix(rep(0, n.outputs), nrow = 1),
    coef.updater = coef.updater,
    bias.updater = bias.updater,
    nonlinearity = nonlinearity,
    prior = prior
  )
  out$resetState(n.minibatch, n.importance.samples)
  
  out
}

defineLayer = function(nonlinearity, size, prior){
  assert_that(inherits(nonlinearity, "nonlinearity"))
  assert_that(inherits(prior, "prior"))
  assert_that(length(size) == 1L)
  list(
    nonlinearity = nonlinearity, 
    size = safe.as.integer(size), 
    prior = prior
  )
}
