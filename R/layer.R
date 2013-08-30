#' @include prior.R
layer = setRefClass(
  Class = "layer",
  fields = list(
    coefficients = "matrix",
    biases = "numeric",
    input = "matrix",
    activation = "matrix",
    output = "matrix",
    error.grad = "matrix",
    llik.grad = "matrix",
    bias.grad = "numeric",
    grad.step = "matrix",
    dim = "integer",
    learning.rate = "numeric",
    momentum = "numeric",
    nonlinearity = "function",
    nonlinearityGrad = "function",
    prior = "prior",
    dataset.size = "numeric",
    dropout = "logical",
    importance.errors = "matrix",
    importance.llik.grads = "array",
    importance.bias.grads = "matrix",
    n.importance.samples = "integer"
  ),
  
  methods = list(
    
    forwardPass = function(input){
      input <<- input
      activation <<- (input %*% coefficients) %plus% biases
      output <<- .self$nonlinearity(activation)
      if(dropout){
        output <<- output * dropoutMask(nrow(output), dim[[2]])
      }
    },
    
    backwardPass = function(incoming.error.grad){
      error.grad <<- nonlinearityGrad(activation) * incoming.error.grad
      llik.grad <<- matrixMultiplyGrad(
        n.in = dim[[1]],
        n.out = dim[[2]],
        error.grad = error.grad,
        input = input
      )
      bias.grad <<- colSums(error.grad)
    },
    
    updateCoefficients = function(){
      grad = -llik.grad + prior$getLogGrad(coefficients) / dataset.size
      grad.step <<- grad * learning.rate + momentum * grad.step
      coefficients <<- coefficients + grad.step
      
      # Hinton suggested the factor of 10 in his "practical guide" for RBMs,
      # if I recall correctly.  The idea is that biases' gradients are easier
      # to estimate reliably, so we can move farther along them.
      # Also, I don't have any momentum for biases at the moment, so this should
      # allow them to keep up better.
      biases <<- biases - bias.grad * learning.rate * 10
    }
  )
)

createLayer = function(
  dim,
  learning.rate,
  momentum,
  prior,
  dataset.size,
  nonlinearity.name,
  n.importance.samples = 1L,
  dropout = FALSE
){
  if(learning.rate > 1 | learning.rate <= 0){
    stop("learning.rate must be greater than 0 and less than or equal to one")
  }
  if(momentum >=1 | momentum < 0){
    stop("momentum cannot be negative and must be less than one")
  }
  
  layer$new(
    coefficients = matrix(0, nrow = dim[[1]], ncol = dim[[2]]),
    biases = rep(0, dim[[2]]),
    grad.step = matrix(0, nrow = dim[[1]], ncol = dim[[2]]),
    dim = dim,
    learning.rate = learning.rate,
    momentum = momentum,
    nonlinearity = get(nonlinearity.name, mode = "function"),
    nonlinearityGrad = get(paste0(nonlinearity.name, "Grad"), mode = "function"),
    prior = prior,
    dataset.size = dataset.size,
    n.importance.samples = n.importance.samples,
    dropout = dropout,
    importance.errors = matrix(NA, nrow = , ncol = n.importance.samples),
    importance.llik.grads = array(
      NA, 
      dim = c(dim[[1]], dim[[2]], n.importance.samples)
    ),
    importance.bias.grads = matrix(
      NA, 
      nrow = dim[[2]], 
      ncol = n.importance.samples
    )
  )
}
