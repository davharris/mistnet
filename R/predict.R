predict.network = function(
  object, 
  newdata, 
  n.importance.samples = 1000L
){
  
  # For some reason, the layers weren't copying by default--only the 
  # reference is being transferred. 
  # As a result, changes to `cpy$layers` were overwriting `object$layers`, 
  # even after exiting the `predict` function.
  # The next couple of lines are designed to prevent that behavior.
  original.layers = lapply(object$layers, function(x) x$copy())
  on.exit({
    object$layers = original.layers
  })
  
  cpy = object$copy(shallow = FALSE)
  cpy$n.importance.samples = n.importance.samples
  # ensure that selectMinibatch triggers its extra routines for changing
  # the number of importance samples.
  # In retrospect, this is a screwy way to control things.
  cpy$selectMinibatch(1L)
  cpy$selectMinibatch(1:nrow(newdata))
  
  # Create an input matrix with the correct dimensions & correct values
  # everywhere that's fixed.
  inputs = cbind(
    newdata, 
    zeros(nrow = nrow(newdata), ncol = n.ranef)
  )
  
  
  for(i in 1:cpy$n.importance.samples){
    inputs[, -(1:ncol(newdata))] = cpy$ranefSample(
      nrow = nrow(newdata), 
      ncol = n.ranef
    )
    cpy$feedForward(
      inputs,
      i
    )
  }

  return(cpy$layers[[length(cpy$layers)]]$outputs)
}
