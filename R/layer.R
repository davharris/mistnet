#' Layer
#'
#' @description A reference class object for one layer of computation in a network object.
#'
#' @details __
#'
#' @field coef.dim a length-two integer vector
#' @field coefficients a matrix of real numbers
#' @field biases a numeric vector
#' @field nonlinearity a \code{nonlinearity} object
#' @field prior a \code{prior} object
#' @field inputs a numeric array
#' @field activations a numeric array
#' @field outputs a numeric array
#' @field error.grads a numeric array
#' @field weighted.bias.grads a numeric vector
#' @field weighted.llik.grads a numeric matrix
#' @field coef.updater an \code{updater} object
#' 
#' @include prior.R
#' @include updater.R
#' @include nonlinearity.R
#' @exportClass
layer = setRefClass(
  Class = "layer",
  fields = list(
    coef.dim = "integer",
    coefficients = "matrix",
    biases = "matrix",
    nonlinearity = "nonlinearity",
    prior = "prior",
    inputs = "array",
    activations = "array",
    outputs = "array",
    error.grads = "array",
    weighted.bias.grads = "numeric",
    weighted.llik.grads = "matrix",
    coef.updater = "updater",
    bias.updater = "updater"
  ),
  
  methods = list(
    
    forwardPass = function(input, sample.num){
      "Update inputs, activations, and outputs for one sample"
      
      if(missing(sample.num)){stop("sample.num is missing in forwardPass")}
      
      inputs[ , , sample.num] <<- input
      activations[ , , sample.num] <<- (input %*% coefficients) %plus% biases
      outputs[ , , sample.num] <<- nonlinearity$f(activations[ , , sample.num])
    },
    
    backwardPass = function(incoming.error.grad, sample.num){
      "Calculate error.grads for one sample"
      
      # Chain rule: multiply incoming error gradient by the nonlinearity's own 
      # gradient.
      nonlinear.grad = nonlinearity$grad(
        activations[ , , sample.num]
      )
      error.grads[ , , sample.num] <<- incoming.error.grad * nonlinear.grad
    },
    
    updateCoefficients = function(
      dataset.size, 
      minibatch.size
    ){
      "Calculate coef.delta and add it to coefficients. Update biases"
      log.prior.grad = prior$getLogGrad(coefficients) / dataset.size
      grad = -weighted.llik.grads / minibatch.size + log.prior.grad
      coef.updater$computeDelta(grad)
      
      coefficients <<- coefficients + coef.updater$delta
      
      bias.grads = weighted.bias.grads / minibatch.size
      bias.updater$computeDelta(bias.grads)
      biases <<- biases - bias.updater$delta
    },
    
    combineSampleGradients = function(weights, n.importance.samples){
      "update weighted.llik.grads and weighted.bias.grads based on importance 
      weights and gradients from backpropagation"
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
      "Reset inputs, activations, outputs, error.grads, and coef.delta to NA"
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
