#' @export
predict.network = function(
  object, 
  newdata, 
  n.importance.samples, 
  return.model = FALSE,
  ...
){
  cpy = object$copy(shallow = FALSE)
  cpy$n.importance.samples = safe.as.integer(n.importance.samples)
  cpy$x = newdata
  cpy$y = matrix(NA, nrow = nrow(newdata), ncol = ncol(object$y))
  cpy$inputs = array(
    NA, 
    dim = c(nrow(newdata), dim(object$inputs)[2], n.importance.samples)
  )
  
  cpy$n.minibatch = 0L
  cpy$selectMinibatch(1:nrow(newdata))
  
  for(i in 1:n.importance.samples){
    cpy$inputs[ , , i] = cbind(
      cpy$x[cpy$minibatch.ids, ], 
      cpy$sampler(nrow = cpy$n.minibatch)
    )
    cpy$feedForward(
      cpy$inputs[ , , i],
      i
    )
  }

  if(return.model){
    cpy
  }else{
    cpy$layers[[length(cpy$layers)]]$outputs
  }
}
