findLogExpectedLik = function(importance.errors, weights){
  # importance.errors should be a matrix. Rows are transects. Columns are samples.
  # Entries should be errors associated with the whole transect (all species).
  # Result should be equivalent to log(mean(exp(-x)))
  # This should make the floating point more accurate when the amount of error is large
  
  if(missing(weights)){
    weights = matrix(
      1 / ncol(importance.errors), 
      nrow = nrow(importance.errors), 
      ncol = ncol(importance.errors)
    )
  }
  sapply(
    1:nrow(importance.errors),
    function(i){
      smallest.error = min(importance.errors[i, ])
      log(
        sum(exp(smallest.error - importance.errors[i, ]) * weights[i, ])
      ) - smallest.error
    }
  )
}
