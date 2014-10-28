#' Construct a mistnet model
#' 
#' This function creates a \code{network} object for fitting a mistnet model.
#' 
#' @param x a \code{numeric} \code{matrix} of predictor variables.  One row per 
#'   record, one column per predictive feature.
#' @param y a \code{matrix} of responses to \code{x}.  One row per record, one 
#'   column per response variable.
#' @param layer.definitions a \code{list} of specifications for each layer in 
#'   the network, as produced by \code{\link{defineLayer}}.
#' @param loss a \code{\link{loss}} object, defining the function for 
#'   optimization to minimize, as well as its gradient.
#' @param updater an \code{\link{updater}} object, specifying how the model 
#'   should move across the likelihood surface (e.g. stochastic gradient descent
#'   or adagrad)
#' @param sampler a \code{\link{sampler}} object, specifying the distribution of
#'   the latent variables
#' @param n.importance.samples an \code{integer}. More samples will take more 
#'   time to compute, but will provide a more precise estimate of the likelihood
#'   gradient.
#' @param n.minibatch an \code{integer} specifying the number of rows to include
#'   in each stochastic estimate of the likelihood gradient.
#' @param training.iterations an \code{integer} number of minibatches to process
#'   before terminating. Defaults to zero so that the user can adjust the 
#'   network before training begins.
#' @param shuffle logical.  Should the data be shuffled after each epoch? 
#'   Defaults to TRUE.
#' @param initialize.biases logical.  Should the network's final layer's biases 
#'   be initialized to nonzero values? If \code{TRUE}, initial values will 
#'   depend on the \code{\link{nonlinearity}} of the final layer. Otherwise, all
#'   values will be zero.
#' @param initialize.weights logical.  Should the weights in each layer be 
#'   initialized automatically? If \code{TRUE}, each \code{\link{layer}}'s
#'   weights will be sampled randomly from their \code{\link{prior}}s.
#'   Otherwise, all values will be zero, which can prevent the network from
#'   learning.
#' @details The \code{mistnet} function produces a \code{\link{network}} object 
#'   that produces a joint distribution over \code{y} given \code{x}. This 
#'   distribution is defined by a stochastic feed-forward neural network (Neal 
#'   1992), which is trained using a variant of backpropagation described in 
#'   Tang and Salakhutdinov (2013) and Harris (2014). During each training 
#'   iteration, model descends the gradient defined by its \code{\link{loss}} 
#'   function, averaged over a number of Monte Carlo samples and a number of 
#'   rows of data.
#'   
#'   A \code{\link{network}} concatenates the predictor variables in \code{x} 
#'   with random variables produced by a \code{\link{sampler}} and passes the 
#'   resulting data vectors through one or more \code{\link{layer}} objects to 
#'   make predictions about \code{y}. The \code{weights} and \code{biases} in 
#'   each \code{\link{layer}} can be trained using the \code{\link{network}}'s 
#'   \code{fit} method (see example below).
#'   
#' @note \code{\link{network}} objects produced by \code{mistnet} are 
#'   \code{\link{ReferenceClasses}}, and behave differently from other R 
#'   objects. In particular, binding a \code{\link{network}} or other reference 
#'   class object to a new variable name will not produce a copy of the original
#'   object, but will instead create a new alias for it.
#' @seealso \code{\link{network}}
#' @seealso \code{\link{layer}}
#' @references Harris, D.J. Building realistic assemblages with a Joint Species 
#'   Distribution Model. BioRxiv preprint. http://dx.doi.org/10.1101/003947
#' @references Neal, R.M. (1992) Connectionist learning of belief networks. 
#'   Artificial Intelligence, 56, 71-113.
#' @references Tang, Y. & Salakhutdinov, R. (2013) Learning Stochastic 
#'   Feedforward Neural Networks. Advances in Neural Information Processing 
#'   Systems 26 (eds & trans C.J.C. Burges), L. Bottou), M. Welling), Z. 
#'   Ghahramani), & K.Q. Weinberger), pp. 530-538.
#' @include prior.R
#' @include nonlinearity.R
#' @import assertthat
#' @examples
#' # 107 rows of fake data
#' x = matrix(rnorm(1819), nrow = 107, ncol = 17) 
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
#' # Fit the model
#' net$fit(iterations = 10)
#' 
#' predict(net, newdata = x, n.importance.samples = 10)
#' @useDynLib mistnet
#' @import Rcpp
#' @import RcppArmadillo
#' @export


mistnet = function(
  x,
  y,
  layer.definitions,
  loss,
  updater,
  sampler = gaussian.sampler(ncol = 10L, sd = 1),
  n.importance.samples = 25,
  n.minibatch = 25,
  training.iterations = 0,
  shuffle = TRUE,
  initialize.biases = TRUE,
  initialize.weights = TRUE
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
  
  
  colnames(net$layers[[net$n.layers]]$weights) = colnames(y)
  dimnames(net$layers[[net$n.layers]]$outputs) = list(NULL, colnames(y), NULL)
  
  if(initialize.biases){
    final.biases = net$layers[[net$n.layers]]$nonlinearity$initializeFinalBiases(y)
    net$layers[[net$n.layers]]$biases[] = final.biases
  }
  if(initialize.weights){
    for(layer in net$layers){
      layer$weights[] = layer$prior$sample(length(layer$weights))
    }
  }
  
  
  net$fit(training.iterations)
  
  return(net)
}
