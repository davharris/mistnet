defineLayer = function(nonlinearity, size, prior){
  assert_that(inherits(nonlinearity, "nonlinearity"))
  assert_that(inherits(prior, "prior"))
  assert_that(length(size) == 1L)
  list(
    nonlinearity = nonlinearity, 
    size = safe.as.integer(size), 
    prior = prior
  )
}

createLayer = function(
  n.inputs,
  n.outputs,
  prior,
  nonlinearity,
  updater,
  n.minibatch,
  n.importance.samples
){
  coef.updater = updater$copy()
  coef.updater$delta = matrix(0, nrow = n.inputs, ncol = n.outputs)
  
  bias.updater = updater$copy()
  bias.updater$delta = matrix(0, nrow = 1, ncol = n.outputs)
  
  out = layer$new(
    coef.dim = c(n.inputs, n.outputs),
    coefficients = matrix(0, nrow = n.inputs, ncol = n.outputs),
    biases = matrix(rep(0, n.outputs), nrow = 1),
    coef.updater = coef.updater,
    bias.updater = bias.updater,
    nonlinearity = nonlinearity,
    prior = prior
  )
  out$resetState(n.minibatch, n.importance.samples)
  
  out
}
