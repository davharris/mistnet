`%plus%` = function(matrix, vector){
  rcpp_add_biases(matrix, vector)
}

crossEntropy = function(y, yhat){
  y * log(yhat) + (1 - y) * log(1 - yhat)
}

sigmoid = function(x) 1 / (1 + exp(-x))

# Note: rectify is defined in src/rectify.cpp