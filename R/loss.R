crossEntropy = function(y, yhat){
  -(y * log(yhat) + (1 - y) * log(1 - yhat))
}

poissonLoss = function(y, yhat){
  - dpois(x = y, lambda = yhat, log = TRUE)
}

squaredLoss = function(y, yhat){
  (y - yhat)^2
}

binomialLoss = function(y, yhat, n){
  -dbinom(x = y, size = n, prob = yhat, log = TRUE)
}

mrfLoss = function(y, yhat, lateral){
  if(!is.matrix(y)){
    y = matrix(y, nrow = 1) # row vector, not column vector
  }

  # The factor of two is because we just use one triangle of `lateral`, not the
  # whole matrix.  See Equation 3 in Osindero and Hinton's
  # "Modeling image patches with a directed hierarchy of Markov random ﬁelds"
  rowSums(crossEntropy(y, yhat)) - sum(lateral * crossprod(y)) / 2
}
