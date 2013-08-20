#' @include prior.R
layer = setRefClass(
  Class = "layer",
  fields = list(
    coefficients = "matrix",
    input = "matrix",
    activation = "matrix",
    output = "matrix",
    llik.grad.estimate = "matrix",
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
      activation <<- input %*% coefficients
      output <<- .self$nonlinearity(activation)
    },
    backwardPass = function(error.gradient){
      llik.grad.estimate <<- matrixMultiplyGrad(
        n.hid = input.dim,
        n.out = output.dim,
        delta = getNonlinearityGrad(),
        h = input
      )
    },
    updateCoefficients = function(){
      grad = llik.gradient.estimate + prior$getLogGrad(coefficients) 
      grad.step <<- grad * learning.rate + momentum * grad.step
      coefficients <<- coefficients + grad.step
    }
  )
)

