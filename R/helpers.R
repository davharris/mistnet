`%plus%` = function(matrix, vector){
  rcpp_add_biases(matrix, vector)
}

crossEntropy = function(y, yhat){
  y * log(yhat) + (1 - y) * log(1 - yhat)
}

sigmoid = function(x){
  if(is.matrix(x)){
    return(sigmoidMatrix(x))
  }else{
    return(sigmoidVector(x))
  }
}

# Note: rectify is defined in src/rectify.cpp

# based on the "josh" function from 
# https://gist.github.com/SChamberlain/3639688
dropoutMask = function(nrow, ncol) {
  out = sample.int(n = 2L, size = nrow * ncol, replace = TRUE) - 1L
  dim(out) = c(nrow, ncol)
  out
}
