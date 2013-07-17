crossEntropyGrad = function(y, yhat){
  - ((y / yhat) - (1 - y) / (1 - yhat))
}

sigmoidGrad = function(s){
  s * (1 - s)
}

passGradThroughSigmoid = function(
  output, 
  previous.output, 
  n.out = ncol(output)
){
  sigmoidGrad(s = output) * repvec(rowSums(previous.output), n.out)
}


rectifiedGrad = function(x){
  x > 0
}