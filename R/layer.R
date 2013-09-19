#' @include prior.R
layer = setRefClass(
  Class = "layer",
  fields = list(
    coef.dim = "integer",
    coefficients = "matrix",
    biases = "numeric",
    nonlinearity = "function",
    nonlinearityGrad = "function",
    prior = "prior",
    inputs = "array",
    activations = "array",
    outputs = "array",
    error.grads = "array",
    weighted.bias.grads = "numeric",
    weighted.llik.grads = "matrix",
    grad.step = "matrix"
  ),
  
  methods = list(
    
    forwardPass = function(input, sample.num){
      if(missing(sample.num)){stop("sample.num is missing in forwardPass")}
      
      inputs[ , , sample.num] <<- input
      activations[ , , sample.num] <<- (input %*% coefficients) %plus% biases
      outputs[ , , sample.num] <<- nonlinearity(activations[ , , sample.num])
    },
    
    backwardPass = function(incoming.error.grad, sample.num){
      # Chain rule: multiply incoming error gradient by the nonlinearity's own 
      # gradient.
      nonlinear.grad = nonlinearityGrad(activations[ , , sample.num])
      error.grads[ , , sample.num] <<- incoming.error.grad * nonlinear.grad
    },
    
    updateCoefficients = function(learning.rate, momentum, dataset.size){
      log.prior.grad = prior$getLogGrad(coefficients) / dataset.size
      grad = -weighted.llik.grads + log.prior.grad
      grad.step <<- grad * learning.rate + momentum * grad.step
      coefficients <<- coefficients + grad.step
      
      # Hinton suggested the factor of 10 in his "practical guide" for RBMs,
      # if I recall correctly.  The idea is that biases' gradients are easier
      # to estimate reliably, so we can move farther along them.
      # Also, I don't have any momentum for biases at the moment, so this should
      # allow them to keep up better.
      biases <<- biases - weighted.bias.grads * learning.rate * 10
    },
    
    combineSampleGradients = function(weights, n.importance.samples){
      weighted.llik.grads <<- zeros(coef.dim[[1]], coef.dim[[2]])
      weighted.bias.grads <<- rep(0, coef.dim[[2]])
      for(i in 1:n.importance.samples){
        partial.error.grad = error.grads[ , , i] * weights[ , i]
        
        partial.llik.grad = matrixMultiplyGrad(
          n.in = coef.dim[[1]],
          n.out = coef.dim[[2]],
          error.grad = partial.error.grad,
          input = inputs[ , , i]
        )
        partial.bias.grad = colSums(partial.error.grad)
        
        weighted.llik.grads <<- weighted.llik.grads + partial.llik.grad
        weighted.bias.grads <<- weighted.bias.grads + partial.bias.grad
      }
    },
    
    resetState = function(minibatch.size, n.importance.samples){
      inputs <<- array(
        NA, 
        c(minibatch.size, coef.dim[[1]], n.importance.samples)
      )
      
      out.array = array(
        NA, 
        c(minibatch.size, coef.dim[[2]], n.importance.samples)
      )
      activations <<- out.array
      outputs <<- out.array
      error.grads <<- out.array
    }
    
  )
)
