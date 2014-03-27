crossEntropyGrad = function(y, yhat){
  (1 - y) / (1 - yhat) - (y / yhat)
}
