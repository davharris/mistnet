crossEntropyGrad = function(y, yhat){
  (1 - y) / (1 - yhat) - (y / yhat)
}

poissonLossGrad = function(y, yhat){
  1 - y / yhat
}

squaredLossGrad = function(y, yhat){
  2 * (yhat - y)
}