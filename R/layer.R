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
    getNonlinearityGrad = "function",
    prior = "prior",
    dataset.size = "numeric"
  ),
  
  methods = list(
    
    forwardPass = function(input){
      input <<- input
      activation <<- (input %*% coefficients) %plus% biases
      output <<- .self$nonlinearity(activation)
    },
    
    backwardPass = function(next.error.grad){
      error.grad <<- `*`(
        getNonlinearityGrad(),
        tcrossprod(next.error.grad, coefficients)
      )
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

