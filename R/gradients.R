crossEntropyGrad = function(y, yhat){
  - ((y / yhat) - (1 - y) / (1 - yhat))
}

sigmoidGrad = function(s){
  s * (1 - s)
}

rectifiedGrad = function(x, b){
  x > b
}
