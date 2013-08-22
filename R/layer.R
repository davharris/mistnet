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
    grad.step = "matrix",
    dim = "integer",
    learning.rate = "numeric",
    momentum = "numeric",
    nonlinearity = "function",
    nonlinearityGrad = "function",
    prior = "prior",
    dataset.size = "numeric",
    dropout = "logical"
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
    },
    
    updateCoefficients = function(){
      grad = llik.grad + prior$getLogGrad(coefficients) / dataset.size
      grad.step <<- grad * learning.rate + momentum * grad.step
      coefficients <<- coefficients + grad.step
      
      # TODO: Update biases
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
    dropout = dropout
  )
}
