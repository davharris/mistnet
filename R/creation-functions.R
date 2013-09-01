mistnet = function(
  x,
  y,
  nonlinearity.names,
  hidden.dims,
  priors,
  learning.rate,
  momentum,
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
    learning.rate = learning.rate,
    momentum = momentum
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
  minibatch.size,
  n.importance.samples
){
  layer$new(
    coef.dim = c(n.inputs, n.outputs),
    coefficients = matrix(0, nrow = n.inputs, ncol = n.outputs),
    biases = rep(0, n.outputs),
    grad.step = matrix(0, nrow = n.inputs, ncol = n.outputs),
    nonlinearity = get(nonlinearity.name, mode = "function"),
    nonlinearityGrad = get(paste0(nonlinearity.name, "Grad"), mode = "function"),
    prior = prior,
    inputs = array(NA, c(minibatch.size, n.inputs, n.importance.samples)),
    activations = array(NA, c(minibatch.size, n.outputs, n.importance.samples)),
    outputs = array(NA, c(minibatch.size, n.outputs, n.importance.samples)),
    error.grads = array(NA, c(minibatch.size, n.outputs, n.importance.samples))
  )
}
