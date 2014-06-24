#' @export
predict.network = function(
  object, 
  newdata, 
  n.importance.samples, 
  return.model = FALSE
){
  
  cpy = object$copy(shallow = FALSE)
  cpy$n.importance.samples = n.importance.samples
  
  cpy$selectMinibatch(1:nrow(newdata))
  
  # Create an input matrix with the correct dimensions & correct values
  # everywhere that's fixed.
  inputs = cbind(
    newdata, 
    zeros(nrow = nrow(newdata), ncol = with(environment(cpy$sampler), ncol))
  )
  
  for(i in 1:cpy$n.importance.samples){
    # Update the unobserved columns, which occur after ncol(newdata)
    inputs[, -(1:ncol(newdata))] = cpy$sampler(
      nrow = nrow(newdata)
    )
    cpy$feedForward(
      inputs,
      i
    )
  }

  if(return.model){
    cpy
  }else{
    cpy$layers[[length(cpy$layers)]]$outputs
  }
}
