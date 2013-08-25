crossEntropyGrad = function(y, yhat){
  - ((y / yhat) - (1 - y) / (1 - yhat))
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

matrixMultiplyGrad = function(n.in, n.out, error.grad, input){
  vapply(
    1:n.out,
    function(i){error.grad[, i] %*% input},
    FUN.VALUE = numeric(n.in)
  )
}
