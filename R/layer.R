#' @include prior.R
#' @include nonlinearity.R
layer = setRefClass(
  Class = "layer",
  fields = list(
    coef.dim = "integer",
    coefficients = "matrix",
    biases = "numeric",
    nonlinearity = "nonlinearity",
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
      outputs[ , , sample.num] <<- nonlinearity$f(activations[ , , sample.num])
    },
    
    backwardPass = function(incoming.error.grad, sample.num){
      # Chain rule: multiply incoming error gradient by the nonlinearity's own 
      # gradient.
      nonlinear.grad = nonlinearity$grad(
        activations[ , , sample.num]
      )
      error.grads[ , , sample.num] <<- incoming.error.grad * nonlinear.grad
    },
    
    updateCoefficients = function(
      learning.rate, 
      momentum, 
      dataset.size, 
      minibatch.size
    ){
      log.prior.grad = prior$getLogGrad(coefficients) / dataset.size
      grad = -weighted.llik.grads / minibatch.size + log.prior.grad
      grad.step <<- grad * learning.rate + momentum * grad.step
      coefficients <<- coefficients + grad.step
      
      # Hinton suggested that biases should have higher learning rates in his
      # "practical guide" for RBMs. The sign is more reliable, so we can move
      # farther.
      # Also, I don't have any momentum for biases at the moment, so this should
      # allow them to keep up better.
      biases <<- biases - weighted.bias.grads * learning.rate / minibatch.size * 10
    },
    
    combineSampleGradients = function(weights, n.importance.samples){
      weighted.llik.grads <<- zeros(coef.dim[[1]], coef.dim[[2]])
      weighted.bias.grads <<- rep(0, coef.dim[[2]])
      for(i in 1:n.importance.samples){
        partial.error.grad = error.grads[ , , i] * weights[ , i]
        
        partial.llik.grad = matrixMultiplyGrad(
          n_out = coef.dim[[2]],
          error_grad = partial.error.grad,
          input_act = inputs[ , , i]
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
