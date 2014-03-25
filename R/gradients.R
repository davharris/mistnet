crossEntropyGrad = function(y, yhat){
  (1 - y) / (1 - yhat) - (y / yhat)
}

sigmoidGrad = function(x){
  s = sigmoid(x)
  s * (1 - s)
}

rectifyGrad = function(x){
  x > 0
}

linearGrad = function(x){
  1
}
