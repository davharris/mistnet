predict.network = function(
  object, 
  newdata, 
  n.importance.samples = 1000L, 
  return.model = FALSE
){
  
  cpy = object$copy(shallow = FALSE)
  cpy$n.importance.samples = n.importance.samples
  
  cpy$selectMinibatch(1:nrow(newdata))
  
  # Create an input matrix with the correct dimensions & correct values
  # everywhere that's fixed.
  inputs = cbind(
    newdata, 
    zeros(nrow = nrow(newdata), ncol = n.ranef)
  )
  
  
  for(i in 1:cpy$n.importance.samples){
    # Update the unobserved columns that occur after ncol(newdata)
    inputs[, -(1:ncol(newdata))] = cpy$ranefSample(
      nrow = nrow(newdata), 
      ncol = n.ranef
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
