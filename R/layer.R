#' @include prior.R
layer = setRefClass(
  Class = "layer",
  fields = list(
    coef.dim = "integer",
    coefficients = "matrix",
    biases = "numeric",
    learning.rate = "numeric",
    momentum = "numeric",
    nonlinearity = "function",
    nonlinearityGrad = "function",
    prior = "prior",
    dataset.size = "numeric",
    n.importance.samples = "integer",
    inputs = "array",
    activations = "array",
    outputs = "array",
    error.grads = "array",
    bias.grads = "matrix",
    llik.grads = "array",
    weighted.error.grads = "matrix",
    weighted.bias.grads = "numeric",
    weighted.llik.grads = "matrix",
    grad.step = "matrix"
  ),
  
  methods = list(
    
    forwardPass = function(input, sample.num){
      inputs[ , , sample.num] <<- input
      activation = (input %*% coefficients) %plus% biases
      activations[ , , sample.num] <<- activation
      outputs[ , , sample.num] <<- nonlinearity(activation)
    },
    
    backwardPass = function(incoming.error.grad, sample.num){
      nonlinear.grad = nonlinearityGrad(activations[ , , sample.num])
      error.grads[ , , sample.num] <<- nonlinear.grad * incoming.error.grad
      
      # Since this step is linear, it might be possible to refactor so I only
      # do it once per update instead of once per importance sample.
      llik.grads[ , , sample.num] <<- matrixMultiplyGrad(
        n.in = coef.dim[[1]],
        n.out = coef.dim[[2]],
        error.grad = error.grads[ , , sample.num],
        input = inputs[ , , sample.num]
      )
      bias.grads[ , sample.num] <<- colSums(error.grads[ , , sample.num])
    },
    
    updateCoefficients = function(){
      grad = -weighted.llik.grads + prior$getLogGrad(coefficients) / dataset.size
      grad.step <<- grad * learning.rate + momentum * grad.step
      coefficients <<- coefficients + grad.step
      
      # Hinton suggested the factor of 10 in his "practical guide" for RBMs,
      # if I recall correctly.  The idea is that biases' gradients are easier
      # to estimate reliably, so we can move farther along them.
      # Also, I don't have any momentum for biases at the moment, so this should
      # allow them to keep up better.
      biases <<- biases - weighted.bias.grad * learning.rate * 10
    },
    averageSampleGradients = function(){
      weighted.bias.grads <<- 0 * weighted.bias.grads
      weighted.llik.grads <<- 0 * weighted.llik.grads
      for(j in 1:n.importance.samples){
        w = weights[ , i]
        weighted.bias.grads <<- weighted.bias.grads + w * bias.grads[ , j]
        # Won't the recycling rule mess this up?
        weighted.llik.grads <<- weighted.llik.grads + w * llik.grads[ , , j]
      }
    }
  )
)
