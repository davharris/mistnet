crossEntropyGrad = function(y, yhat){
  - ((y / yhat) - (1 - y) / (1 - yhat))
}

sigmoidGrad = function(s){
  s * (1 - s)
}

rectifiedGrad = function(x){
  x > 0
}

matrixMultiplyGrad = function(n.in, n.out, error.grad, input){
  -t(
    vapply(
      1:n.out,
      function(i){error.grad[, i] %*% input},
      FUN.VALUE = numeric(n.in)
    )
  )
}
