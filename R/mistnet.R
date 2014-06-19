#' Construct a mistnet model
#' 
#' This function creates a \code{network} object for fitting a mistnet model.
#' 
#' @param x a \code{numeric} \code{matrix} of predictor variables.  One row
#'  per example, one column per predictive feature.
#' @param y a \code{matrix} of responses to \code{x}.  One row per example, one
#'  column per response variable.
#' @param layer.definitions a \code{list} of specifications for each layer in
#'  the network, as produced by \code{defineLayer}.
#' @param loss a \code{loss} object, defining the function for optimization to 
#' minimize, as well as its gradient
#' @param updater an \code{updater} object, specifying how the model should move
#'  across the likelihood surface (e.g. stochastic gradient descent or adagrad)
#' @param sampler a \code{sampler} object, specifying the distribution of the
#'  latent variables
#' @param n.importance.samples an \code{integer}. More samples will take more time
#'  to compute, but will provide a more precise estimate of the likelihood gradient.
#' @param n.minibatch an \code{integer} specifying the number of rows to include
#'  in each stochastic estimate of the likelihood gradient.
#' @param training.iterations an \code{integer} number of minibatches to process
#'  before terminating.
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
