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
#'  before terminating. Currently, it is best practice to leave this value at 0
#'  and manually initialize the model's coefficients before beginning training
#' @seealso \code{\link{network}}
#' @include prior.R
#' @include nonlinearity.R
#' @import assertthat
#' @examples
#' # 107 rows of fake data
#' x = matrix(rnorm(1819), ncol = 17, nrow = 107) 
#' y = dropoutMask(107, 14)
#' 
#' # Create the network object
#' net = mistnet(
#'   x = x,
#'   y = y,
#'   layer.definitions = list(
#'     defineLayer(
#'       nonlinearity = rectify.nonlinearity(), 
#'       size = 30, 
#'       prior = gaussian.prior(mean = 0, sd = 0.1)
#'     ),
#'     defineLayer(
#'       nonlinearity = rectify.nonlinearity(), 
#'       size = 12, 
#'       prior = gaussian.prior(mean = 0, sd = 0.1)
#'     ),
#'     defineLayer(
#'       nonlinearity = sigmoid.nonlinearity(), 
#'       size = ncol(y), 
#'       prior = gaussian.prior(mean = 0, sd = 0.1)
#'     )
#'   ),
#'   loss = bernoulliLoss(),
#'   updater = adagrad.updater(learning.rate = .01),
#'   sampler = gaussian.sampler(ncol = 10L, sd = 1),
#'   n.importance.samples = 30,
#'   n.minibatch = 10,
#'   training.iterations = 0
#' )
#' 
#' # Currently, mistnet does not initialize the coefficients automatically.
#' # This gets it started with nonzero values.
#' for(layer in net$layers){
#'   layer$coefficients[ , ] = rnorm(length(layer$coefficients), sd = .01)
#' }
#' 
#' # Fit the model
#' net$fit(iterations = 10)
#' @useDynLib mistnet
#' @import Rcpp
#' @import RcppArmadillo
#' @export


mistnet = function(
  x,
  y,
  layer.definitions,
  loss,
  updater = adagrad.updater(learning.rate = .01),
  sampler = gaussian.sampler(ncol = 10L, sd = 1),
  n.importance.samples = 30,
  n.minibatch = 10,
  training.iterations = 0,
  shuffle = TRUE,
  initialize.biases = FALSE,
  initialize.coefficients = FALSE
){  
  assert_that(is.matrix(x))
  assert_that(is.matrix(y))
  assert_that(nrow(x) == nrow(y))
  dataset.size = nrow(x)
  
  assert_that(n.minibatch > 0)
  assert_that(n.minibatch <= dataset.size)
  
  n.layers = length(layer.definitions)
  assert_that(n.layers > 0)
  
  # Input size followed by all the output sizes, in order
  network.dims = c(
    ncol(x) + sampler$ncol, 
    sapply(layer.definitions, function(x) x$size)
  )
  
  if(network.dims[length(network.dims)] != ncol(y)){
    stop("The number of outputs in the last layer must equal the number of columns in y")
  }
  
  net = network$new( 
    x = x,
    y = y,
    layers = lapply(
      1:n.layers,
      function(i){
        createLayer(
          n.inputs = network.dims[[i]],
          n.outputs = network.dims[[i + 1]],
          prior = layer.definitions[[i]]$prior,
          nonlinearity = layer.definitions[[i]]$nonlinearity,
          updater = updater,
          n.minibatch = safe.as.integer(n.minibatch),
          n.importance.samples = safe.as.integer(n.importance.samples)
        )
      }
    ),
    n.layers = n.layers,
    dataset.size = dataset.size,
    row.selector = row.selector(
      rows = if(shuffle){
        sample.int(dataset.size)
      }else{
        seq_len(dataset.size)
      }, 
      shuffle = shuffle,
      dataset.size = safe.as.integer(dataset.size),
      n.minibatch = safe.as.integer(n.minibatch),
      completed.epochs = 0L,
      minibatch.ids = rep(0L, n.minibatch),
      pointer = 1L
    ),
    n.importance.samples = safe.as.integer(n.importance.samples),
    loss = loss$loss,
    lossGradient = loss$grad,
    sampler = sampler,
    completed.iterations = 0L,
    debug = FALSE
  ) 
  net$inputs = array(
    0, 
    c(
      net$row.selector$n.minibatch, 
      ncol(net$x) + net$sampler$ncol, 
      net$n.importance.samples
    )
  )
  
  
  colnames(net$layers[[net$n.layers]]$coefficients) = colnames(y)
  dimnames(net$layers[[net$n.layers]]$outputs) = list(NULL, colnames(y), NULL)
  
  if(initialize.biases){
    final.biases = net$layers[[net$n.layers]]$nonlinearity$initializeFinalBiases(y)
    net$layers[[net$n.layers]]$biases[] = final.biases
  }
  if(initialize.coefficients){
    for(layer in net$layers){
      layer$coefficients[] = layer$prior$sample(length(layer$coefficients))
    }
  }
  
  
  net$fit(training.iterations)
  
  return(net)
}
