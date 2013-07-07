crossEntropyGrad = function(y, yhat){
  -y / yhat
}

sigmoidGrad = function(x){
  x * (1 - x)
}

rectifiedGrad = function(x, b){
  x > b
}
