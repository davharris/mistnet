#' Layer
#'
#' @description A reference class object for one layer of computation in a network object.
#'
#' @details __
#'
#' @field coef.dim a length-two integer vector
#' @field coefficients a matrix of real numbers
#' @field biases a numeric vector containing the intercept for each node
#' @field nonlinearity a \code{\link{nonlinearity}} object
#' @field prior a \code{\link{prior}} object
#' @field inputs a numeric array with the input activity to each
#'  node in response to each example, for each Monte Carlo sample
#' @field outputs a numeric array with the transformed activations for each 
#'  node in response to each example, for each Monte Carlo sample
#' @field error.grads a numeric array
#' @field weighted.bias.grads a numeric vector
#' @field weighted.llik.grads a numeric matrix
#' @field coef.updater an \code{\link{updater}} object
#' 
#' @include prior.R
#' @include updater.R
#' @include nonlinearity.R
#' @seealso \code{\link{network}}
#' @export
layer = setRefClass(
  Class = "layer",
  fields = list(
    coef.dim = "integer",
    coefficients = "matrix",
    biases = "matrix",
    nonlinearity = "nonlinearity",
    prior = "prior",
    inputs = "array",
    outputs = "array",
    error.grads = "array",
    weighted.bias.grads = "numeric",
    weighted.llik.grads = "matrix",
    coef.updater = "updater",
    bias.updater = "updater"
  ),
  
  methods = list(
    
    forwardPass = function(input, sample.num){
      "Update inputs and outputs for one sample"
      
      if(missing(sample.num)){stop("sample.num is missing in forwardPass")}
      
      inputs[ , , sample.num] <<- (input %*% coefficients) %plus% biases
      outputs[ , , sample.num] <<- nonlinearity$f(inputs[ , , sample.num])
    },
    
    backwardPass = function(incoming.error.grad, sample.num){
      "Calculate error.grads for one sample"
      
      # Chain rule: multiply incoming error gradient by the nonlinearity's own 
      # gradient.
      nonlinear.grad = nonlinearity$grad(
        inputs[ , , sample.num]
      )
      error.grads[ , , sample.num] <<- incoming.error.grad * nonlinear.grad
    },
    
    updateCoefficients = function(
      dataset.size, 
      n.minibatch
    ){
      "Calculate coef.delta and add it to coefficients. Update biases"
      log.prior.grad = prior$getLogGrad(coefficients) / dataset.size
      grad = weighted.llik.grads / n.minibatch - log.prior.grad
      
      coef.updater$computeDelta(grad)
      
      coefficients <<- coefficients + coef.updater$delta
      
      bias.updater$computeDelta(weighted.bias.grads / n.minibatch)
      biases <<- biases + bias.updater$delta
    },
    
    combineSampleGrads = function(inputs, weights, n.importance.samples){
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
    
    resetState = function(n.minibatch, n.importance.samples){
      "Reset inputs, outputs, and error.grads to NA;
      alter the minibatch size and number of importance samples if desired"
      
      out.array = array(
        NA, 
        c(n.minibatch, coef.dim[[2]], n.importance.samples)
      )
      inputs <<- out.array
      outputs <<- out.array
      error.grads <<- out.array
    }
    
  )
)
