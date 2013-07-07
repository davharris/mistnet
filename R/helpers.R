`%plus%` = function(matrix, vector){
  rcpp_add_biases(matrix, vector)
}

crossEntropy = function(y, yhat){
  - y * log(yhat)
}

sigmoid = function(x) 1 / (1 + exp(-x))
