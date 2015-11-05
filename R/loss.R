#' Loss functions
#' 
#' Loss functions are minimized during model training. \code{loss} objects contain
#'  a \code{loss} function as well as a \code{grad} function, specifying the gradient.
#'  \code{loss} classes like the negative binomial can also store parameters that can be
#'  updated during training.
#' 
#' @rdname loss
#' @export loss
#' @exportClass loss
loss = setRefClass(
  Class = "loss",
  fields = list(
    updater = "updater"
  ),
  methods = list(
    loss = function(y, yhat) stop({"loss function not defined for this loss class"}),
    grad = function(y, yhat){stop("gradient not defined for this loss class")},
    update = function(y, yhat){
      # (Do nothing for most loss classes)
    }
  )
)


#' \code{bernoulliLoss}:
#' cross-entropy for 0-1 data. Equal to 
#'   \code{-(y * log(yhat) + (1 - y) * log(1 - yhat))}
#' @rdname loss
#' @include cross-entropy.R
#' @export bernoulliLoss
#' @exportClass bernoulliLoss
bernoulliLoss = setRefClass(
  Class = "bernoulliLoss",
  contains = "loss",
  methods = list(
    loss = crossEntropy,
    grad = crossEntropyGrad
  )
)


#' \code{bernoulliRegLoss}: cross-entropy loss, regularized by a 
#'   beta-distributed prior.
#' Note that \code{a} and \code{b} are not 
#' @param a the \code{a} shape parameter in \code{\link{dbeta}}
#' @param b the \code{b} shape parameter in \code{\link{dbeta}}
#' @rdname loss
#' @include cross-entropy.R
#' @export bernoulliRegLoss
#' @exportClass bernoulliRegLoss
#' 
bernoulliRegLoss = setRefClass(
  Class = "bernoulliRegLoss",
  fields = list(
    a = "any.numeric",
    b = "any.numeric"
  ),
  contains = "loss",
  methods = list(
    loss = function(y, yhat){crossEntropyReg(y = y, yhat = yhat, a = a, b = b)},
    grad = function(y, yhat){crossEntropyRegGrad(y = y, yhat = yhat, a = a, b = b)}
  )
)

#' \code{poissonLoss}: loss based on the Poisson likelihood.  
#'   See \code{\link{dpois}}
#' @rdname loss
#' @export poissonLoss
#' @exportClass poissonLoss
poissonLoss = setRefClass(
  Class = "poissonLoss",
  contains = "loss",
  methods = list(
      loss = function(y, yhat){
        - dpois(x = y, lambda = yhat, log = TRUE)
      },
      grad = function(y, yhat){
        1 - y / yhat
      }
  )
)

#' \code{nbLoss}: loss based on the negative binomial likelihood
#'   See \code{\link{dnbinom}}
#' @rdname loss
#' @export nbLoss
#' @exportClass nbLoss
nbLoss = setRefClass(
  Class = "nbLoss",
  contains = "loss",
  fields = list(
    log_size = "numeric"
  ),
  methods = list(
    loss = function(y, yhat){
      - dnbinom(x = y, size = exp(log_size), mu = yhat, log = TRUE)
    },
    grad = function(y, yhat){
      - exp(log_size) * (y - yhat) / (yhat * (exp(log_size) + yhat))
    }
  )
)


#' \code{squaredLoss}: Squared error, for linear models
#' @rdname loss
#' @export squaredLoss
#' @exportClass squaredLoss
squaredLoss = setRefClass(
  Class = "squaredLoss",
  contains = "loss",
  methods = list(
    loss = function(y, yhat){
      (y - yhat)^2
    },
    grad = function(y, yhat){
      2 * (yhat - y)
    }
  )
)

normalLoss = setRefClass(
  Class = "normalLoss",
  contains = "loss",
  fields = list(sigma = "numeric"),
  methods = list(
    
    update = function(y, yhat){
      
      (yhat^2 - sigma^2 + y^2 -2 *yhat * y)/sigma^3
    }
  )
)


#' \code{binomialLoss}: loss for binomial responses. 
#' @param n specifies the number of Bernoulli trials (\code{size} in 
#'   \code{\link{dbinom}})
#' @rdname loss
#' @export binomialLoss
#' @exportClass binomialLoss
binomialLoss = setRefClass(
  Class = "binomialLoss",
  contains = "loss",
  fields = list(
    n = "integer"
  ),
  methods = list(
    loss = function(y, yhat){
      # Should inherit `n` from the parent environment
      -dbinom(x = y, size = n, prob = yhat, log = TRUE)
    },
    grad = function(y, yhat, n){
      (n * yhat - y) / (yhat - yhat^2)
    }
  )
)

mrfLoss = setRefClass(
  Class = "mrfLoss",
  contains = "loss",
  methods = list(
    loss = function(y, yhat, lateral){
      if(is.vector(y)){
        y = matrix(y, nrow = 1) # row vector, not column vector
      }
      cross.entropy.loss = rowSums(crossEntropy(y, yhat))
      
      # The factor of two is because we just use one triangle of `lateral`, 
      # not the whole matrix.  See Equation 3 in Osindero and Hinton's
      # "Modeling image patches with a directed hierarchy of Markov random fields"
      cross.entropy.loss - sapply(
        1:nrow(y),
        function(i){
          sum(lateral * crossprod(y[i, , drop = FALSE])) / 2
        }
      )
    },
    grad = crossEntropyGrad
  )
)




