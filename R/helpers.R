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
#' @param nrow the number of rows desired
#' @param ncol the number of columns desired
#' @export
dropoutMask = function(n.row, n.col) {
  out = sample.int(n = 2L, size = n.row * n.col, replace = TRUE) - 1L
  dim(out) = c(n.row, n.col)
  out
}

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
