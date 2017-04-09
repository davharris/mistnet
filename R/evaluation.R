# Result should be equivalent to log(mean(exp(x))).
# The extra work is to deal with floating point error, overflow, and weights.
# Oddly enough, floating point precision doesn't seem to cause problems.
logMeanExp = function(x, weights){
  if(missing(weights)){
    weights = rep(1/length(x), length(x))
  }
  assert_that(all(weights >= 0))
  weights = weights / sum(weights)
  
  # Rescale so the largest log-likelihoods values are near zero, then undo the
  # rescaling
  biggest = max(x)
  log(sum(exp(x - biggest) * weights)) + biggest
}

#' Evaluate a network object's predictions on newdata against observed y.
#' 
#' Estimate the model's log-likelihood on data held out of sample.
#' Rather than storing a huge number of sample predictions in memory, we can 
#' do this in #' batches of a specified size and just work with the scalar likelihood.
#' 
#' @param object a \code{network} object
#' @param newdata a matrix of predictor variables.
#' @param y a matrix of response variables
#' @param loss a \code{loss} function (e.g. from a \code{loss} object), assumed
#'   to be a *negative* log-likelihood
#' @param batches the number of importance sampling batches to perform
#' @param batch.size the number of importance samples per batch
#' @export
importanceSamplingEvaluation = function(
  object, 
  newdata, 
  y, 
  loss,
  batches, 
  batch.size
){
  
  logliks = replicate(
    batches,{
      predictions = predict(
        object, 
        newdata, 
        n.importance.samples = safe.as.integer(batch.size)
      )
      
      loglik = apply(
        predictions, 
        3, 
        function(x){
          -rowSums(loss(y = y, yhat = x))
        }
      )
      
      loglik
    },
    simplify = FALSE
  )
  
  apply(do.call(cbind, logliks), 1, logMeanExp)
}


