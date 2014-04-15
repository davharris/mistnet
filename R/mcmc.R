metropolisStep = function(object, y, proposal.scale = 1/4){
  # One Metropolis step for random effects
  # One independent chain per row of data.
  # Only use the first slice of arrays (instead of all n.importance.samples)
  
  # Save old material for comparison with proposed values.
  # Calculating old.loss will be redundant sometimes (if only Metropolis is
  # used), but will be important if Metropolis is mixed with other 
  # transitions.
  old.latent = object$layers[[1]]$inputs[ , -(1:ncol(object$x)), 1]
  object$feedForward(
    cbind(
      object$x[object$minibatch.ids, ], 
      old.latent
    ),
    1
  )
  old.loss = -rowSums(dnorm(old.latent, mean = 0, sd = 1, log = TRUE)) + 
    rowSums(
      object$loss(
        y = y, 
        yhat = object$layers[[object$n.layers]]$outputs[, , 1]
      ) 
    )
  
  # Make predictions for proposals in all rows
  latent = old.latent + proposal.scale * object$ranefSample(
    nrow = object$minibatch.size, 
    ncol = object$n.ranef
  )
  
  # The "ones" are because we're using the first slice only.
  object$feedForward(
    cbind(
      object$x[object$minibatch.ids, ], 
      latent
    ),
    1
  )
  
  # Evaluate proposals
  loss = -rowSums(dnorm(latent, mean = 0, sd = 1, log = TRUE)) + 
    rowSums(
      object$loss(
        y = y, 
        yhat = object$layers[[object$n.layers]]$outputs[, , 1]
      ) 
    )
  
  acceptance.ratio = exp(old.loss - loss)
  accept = runif(length(acceptance.ratio)) < acceptance.ratio
  
  # Replace rejected proposals with their previous values
  object$layers[[1]]$inputs[!accept, -(1:ncol(object$x)), 1] = old.latent[!accept, ]
  
  # Return post-rejection loss
  ifelse(accept, loss, old.loss)
}
