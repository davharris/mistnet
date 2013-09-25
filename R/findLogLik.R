findLogExpectedLik = function(importance.errors, weights){
  # Result should be equivalent to log(mean(exp(-x)))
  # This should make the floating point more when the amount of error is large
  
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
        sum(exp(smallest.error - importance.errors[i, ]) * weights[i, ]) * 
          exp(-smallest.error)
      )
    }
  )
}
