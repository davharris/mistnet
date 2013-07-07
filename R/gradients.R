crossEntropyGrad = function(y, yhat){
  -y / yhat
}

sigmoidGrad = function(s){
  s * (1 - s)
}

rectifiedGrad = function(x, b){
  x > b
}
