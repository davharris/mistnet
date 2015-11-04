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
