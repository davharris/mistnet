`%plus%` = function(matrix, vector){
  rcpp_add_biases(matrix, vector)
}

rectify = function(x) pmax(x, 0)
