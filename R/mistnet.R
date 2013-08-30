mistnet = function(
  x,
  y,
  output.distribution,
  nonlinearity.names,
  hidden.dims,
  priors,
  learning.schedule,
  momentum.schedule,
  n.ranef,
  n.importance.samples,
  minibatch.size = 50,
  training.iterations
){
  n.layers = length(nonlinearity.names)
  stopifnot(length(priors) == n.layers)
  stopifnot((length(hidden.dims) + 1L) == n.layers)
  
  stopifnot(hidden.dims == as.integer(hidden.dims))
  stopifnot(n.ranef == as.integer(n.ranef))
  stopifnot(n.ranef == as.integer(n.ranef))
  
  
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
          dim = network.dims[c(i, i + 1)],
          learning.rate,
          momentum,
          prior = priors[[i]],
          dataset.size = dataset.size,
          nonlinearity.name = nonlinearity.names[[i]],
          dropout = FALSE
        )
      }
    ),
    n.layers = n.layers,
    minibatch.size = minibatch.size,
    n.importance.samples = n.importance.samples,
    loss = output.distribution$loss,
    lossGradient = output.distribution$lossGrad,
    ranefSample = gaussianRanefSample(),
    n.ranef = n.ranef 
  ) 
  
  # Coefficients can't all start at zero!
  # Final layer's biases shouldn't be zero either!
  
  net$fit(training.iterations)
  
  return(net)
}
