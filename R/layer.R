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
    coef.grad = "matrix",
    grad.step = "matrix",
    input.dim = "integer",
    output.dim = "integer",
    learning.rate = "numeric",
    momentum = "numeric",
    nonlinearity = "function",
    getNonlinearityGrad = "function",
    prior = "prior"
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
      coef.grad <<- matrixMultiplyGrad(
        n.hid = input.dim,
        n.out = output.dim,
        delta = error.gradient,
        h = input
      )
    },
    updateCoefficients = function(){
      grad = coef.grad + prior$getLogGrad(coefficients) 
      grad.step <<- grad * learning.rate + momentum * grad.step
      coefficients <<- coefficients + grad.step
    }
  )
)

