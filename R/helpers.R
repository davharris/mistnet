`%plus%` = function(matrix, vector){
  rcpp_add_biases(matrix, vector)
}

crossEntropy = function(y, yhat){
  -(y * log(yhat) + (1 - y) * log(1 - yhat))
}


# based on the "josh" function from 
# https://gist.github.com/SChamberlain/3639688
dropoutMask = function(nrow, ncol) {
  out = sample.int(n = 2L, size = nrow * ncol, replace = TRUE) - 1L
  dim(out) = c(nrow, ncol)
  out
}

zeros = function(nrow, ncol){
  matrix(0, nrow = nrow, ncol = ncol)
}

weighImportance = function(importance.errors){
  unscaled.weights = t(apply(
    importance.errors, 
    1,
    function(x) exp(min(x) - x)
  ))
  unscaled.weights / rowSums(unscaled.weights)
}
