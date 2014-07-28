# Confirm that coercing to integer doesn't change the value
safe.as.integer = function(x){
  assert_that(are_equal(x, as.integer(x)))
  as.integer(x)
}

`%plus%` = function(matrix, vector){
  rcpp_add_biases(matrix, vector)
}

#' Produce a dropout mask
#' 
#' Produces a binary matrix, with each element independently sampled as 1 or 0
#' with probabilty 0.5.
#' Code is based on the "josh" function from 
#' https://gist.github.com/sckott/3639688
#' @param n.row the number of rows desired
#' @param n.col the number of columns desired
#' @param prob the probability that an element of the matrix is one rather than zero
#' @export
dropoutMask = function(n.row, n.col, prob = 0.5) {
  out = - 1L +  sample.int(n = 2L, 
                           size = n.row * n.col, 
                           replace = TRUE, 
                           prob = c(1-prob, prob)
  )
  dim(out) = c(n.row, n.col)
  out
}


#' Produce a matrix of zeros
#' 
#' @param n.row the number of rows desired
#' @param n.col the number of columns desired
#' @export
zeros = function(n.row, n.col){
  matrix(0, nrow = n.row, ncol = n.col)
}

weighImportance = function(importance.errors){
  unscaled.weights = t(apply(
    importance.errors, 
    1,
    function(x) exp(min(x) - x)
  ))
  unscaled.weights / rowSums(unscaled.weights)
}
