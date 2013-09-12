findLogExpectedLik = function(importance.errors){
  apply(
    # Result should be equivalent to log(mean(exp(-x)))
    # This should make the floating point more when the amount of error is large
    
    importance.errors, 
    1,
    function(x){
      smallest.error = min(x)
      log(mean(exp(smallest.error - x)) * exp(-smallest.error))
    }
  )
}
