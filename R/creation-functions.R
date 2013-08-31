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


createLayer = function(
  coef.dim,
  learning.rate,
  momentum,
  prior,
  dataset.size,
  nonlinearity.name,
  n.importance.samples
){
  
  if(learning.rate > 1 | learning.rate < 0){
    stop("learning.rate must be between 0 and 1 (inclusive)")
  }
  if(learning.rate == 0){
    warning("learning.rate is zero: training will not adjust the coefficients")
  }
  if(momentum >= 1 | momentum < 0){
    stop("momentum cannot be negative and must be less than one")
  }
  
  layer$new(
    coefficients = matrix(0, nrow = coef.dim[[1]], ncol = coef.dim[[2]]),
    biases = rep(0, dim[[2]]),
    grad.step = matrix(0, nrow = coef.dim[[1]], ncol = coef.dim[[2]]),
    coef.dim = coef.dim,
    learning.rate = learning.rate,
    momentum = momentum,
    nonlinearity = get(nonlinearity.name, mode = "function"),
    nonlinearityGrad = get(paste0(nonlinearity.name, "Grad"), mode = "function"),
    prior = prior,
    dataset.size = dataset.size,
    n.importance.samples = n.importance.samples,
    llik.grads = array(
      NA, 
      dim = c(coef.dim[[1]], coef.dim[[2]], n.importance.samples)
    ),
    bias.grads = matrix(
      NA, 
      nrow = coef.dim[[2]], 
      ncol = n.importance.samples
    )
  )
}
