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

passGradThroughSigmoid = function(n.hid, n.out, y, o, h){
  -t(
    vapply(
      1:n.out,
      function(i){
        (crossEntropyGrad(y = y, yhat = o)[, i] * sigmoidGrad(s = o)[, i]) %*% h
      },
      FUN.VALUE = numeric(n.hid)
    )
  )
}