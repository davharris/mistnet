mistnet = function(
  x,
  y,
  nonlinearity.names,
  hidden.dims,
  priors,
  updater.name = "sgd",
  updater.arguments = list(learning.rate = .001, momentum = .9),
  n.ranef,
  ranefSample,
  n.importance.samples,
  minibatch.size,
  training.iterations,
  loss,
  lossGrad
){
  n.layers = length(nonlinearity.names)
  stopifnot(length(priors) == n.layers)
  stopifnot((length(hidden.dims) + 1L) == n.layers)
  
  stopifnot(hidden.dims == as.integer(hidden.dims))
  stopifnot(n.ranef == as.integer(n.ranef))
  stopifnot(minibatch.size == as.integer(minibatch.size))
  # etc.  Probably worth writing a function for this...
  
  
  network.dims = as.integer(
    c(ncol(x) + n.ranef, hidden.dims, ncol(y))
  )
  
  stopifnot(nrow(x) == nrow(y))
  dataset.size = nrow(x)
  
  net = network$new( 
    x = x,
    y = y,
    layers = lapply(
      1:n.layers,
      function(i){
        createLayer(
          n.in = network.dims[[i]],
          n.out = network.dims[[i + 1]],
          prior = priors[[i]],
          nonlinearity.name = nonlinearity.names[[i]],
          updater.name = updater.name,
          updater.arguments = updater.arguments,
          minibatch.size = minibatch.size,
          n.importance.samples = n.importance.samples
        )
      }
    ),
    n.layers = n.layers,
    dataset.size = dataset.size,
    minibatch.size = minibatch.size,
    n.importance.samples = n.importance.samples,
    loss = loss,
    lossGradient = lossGrad,
    ranefSample = ranefSample,
    n.ranef = n.ranef,
    completed.iterations = 0L
  ) 
  
  # Coefficients can't all start at zero! Perhaps sample coefficients from their
  # prior?
  # Final layer's biases shouldn't be zero either!
  
  net$fit(training.iterations)
  
  return(net)
}


createLayer = function(
  n.inputs,
  n.outputs,
  prior,
  nonlinearity.name,
  updater.name,
  updater.arguments,
  minibatch.size,
  n.importance.samples
){
  out = layer$new(
    coef.dim = c(n.inputs, n.outputs),
    coefficients = matrix(0, nrow = n.inputs, ncol = n.outputs),
    biases = matrix(rep(0, n.outputs), nrow = 1),
    coef.updater = do.call(
      new,
      c(
        list(
          Class = paste(updater.name, "updater", sep = "."),
          delta = matrix(0, nrow = n.inputs, ncol = n.outputs)
        ),
        updater.arguments
      )
    ),
    bias.updater = do.call(
      new,
      c(
        list(
          Class = paste(updater.name, "updater", sep = "."),
          delta = matrix(0, nrow = 1, ncol = n.outputs)
        ),
        updater.arguments
      )
    ),
    nonlinearity = new(
      paste(nonlinearity.name, "nonlinearity", sep = ".")
    ),
    prior = prior
  )
  out$resetState(minibatch.size, n.importance.samples)
  
  out
}
