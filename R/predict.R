#' @import progress
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
  
  cpy$row.selector$n.minibatch = 0L
  cpy$selectMinibatch(1:nrow(newdata))
  
  pb = progress_bar$new(total = n.importance.samples)
  
  for(i in 1:n.importance.samples){
    pb$tick()
    cpy$inputs[ , , i] = cbind(
      cpy$x[cpy$row.selector$minibatch.ids, ], 
      cpy$sampler$sample(nrow = cpy$row.selector$n.minibatch)
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
