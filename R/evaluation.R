# Result should be equivalent to log(mean(exp(x))).
# The extra work is to deal with floating point error, overflow, and weights.
# Oddly enough, floating point precision doesn't seem to cause problems.
logMeanExp = function(x, weights){
  if(missing(weights)){
    weights = rep(1/length(x), length(x))
  }
  assert_that(all(weights > 0))
  weights = weights / sum(weights)
  
  # Rescale so the largest log-likelihoods values are near zero, then undo the
  # rescaling
  biggest = max(x)
  log(sum(exp(x - biggest) * weights)) + biggest
}

# Evaluate a network object's predictions on newdata against observed y.
# Rather than storing a huge number of samples in memory, we can do this in
# batches of a specified size.
importanceSamplingEvaluation = function(object, newdata, y, batches, batch.size){
  logliks = replicate(
    batches,{
      predictions = predict(
        object, 
        newdata, 
        n.importance.samples = safe.as.integer(batch.size)
      )
      
      apply(
        predictions, 
        3, 
        function(x){
          -rowSums(object$loss(y = y, yhat = x))
        }
      )
    },
    simplify = FALSE
  )
  
  apply(do.call(cbind, logliks), 1, logMeanExp)
}


