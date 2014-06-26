#' @export
bernoulliLoss = function(){
  structure(
    list(
      loss = crossEntropy,
      grad = crossEntropyGrad
    ),
    class = "loss"
  )
}

#' @export
bernoulliRegLoss = function(a, b = a){
  structure(
    list(
      loss = function(y, yhat){crossEntropyReg(y = y, yhat = yhat, a = a, b = b)},
      grad = function(y, yhat){crossEntropyRegGrad(y = y, yhat = yhat, a = a, b = b)}
    ),
    class = "loss"
  )
}


#' @export
poissonLoss = function(){
  structure(
    list(
      loss = function(y, yhat){
        - dpois(x = y, lambda = yhat, log = TRUE)
      },
      grad = function(y, yhat){
        1 - y / yhat
      }
    ),
    class = "loss"
  )
}

#' @export
squaredLoss = function(){
  structure(
    list(
      loss = function(y, yhat){
        (y - yhat)^2
      },
      grad = function(y, yhat){
        2 * (yhat - y)
      }
    ),
    class = "loss"
  )
  
}

# n is the size argument for dbinom
#' @export
binomialLoss = function(n){
  
  n = safe.as.integer(n)
  
  structure(
    list(
      loss = function(y, yhat){
        # Should inherit `n` from the parent environment
        -dbinom(x = y, size = n, prob = yhat, log = TRUE)
      },
      grad = function(y, yhat, n){
        (n * yhat - y) / (yhat - yhat^2)
      }
    ),
    class = "loss"
  )
}

#' @export
mrfLoss = function(){
  structure(
    list(
      loss = function(y, yhat, lateral){
        if(is.vector(y)){
          y = matrix(y, nrow = 1) # row vector, not column vector
        }
        cross.entropy.loss = rowSums(crossEntropy(y, yhat))
        
        # The factor of two is because we just use one triangle of `lateral`, 
        # not the whole matrix.  See Equation 3 in Osindero and Hinton's
        # "Modeling image patches with a directed hierarchy of Markov random ﬁelds"
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
}




# Cross Entropy functions -------------------------------------------------

crossEntropy = function(y, yhat){
  -(y * log(yhat) + (1 - y) * log(1 - yhat))
}
crossEntropyGrad = function(y, yhat){
  (1 - y) / (1 - yhat) - (y / yhat)
}

# Regularized cross entropy: includes a beta prior that the predictions won't
# be less than (a-1) when a==b.  Rules out things like billion-to-one odds if
# a==1 + 1E-7
crossEntropyReg = function(y, yhat, a, b){
  -(y * log(yhat) + (1 - y) * log(1 - yhat)) - dbeta(yhat, a, b, log = TRUE)
}
crossEntropyRegGrad = function(y, yhat, a, b = a){
  (1 - y) / (1 - yhat) - (y / yhat) - ((a - 1)/yhat + (b - 1)/(yhat - 1))
}
