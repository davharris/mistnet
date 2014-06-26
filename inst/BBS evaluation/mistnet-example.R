load("birds.Rdata")
env = scale(x[, 1:8])
devtools::load_all()

# Best model in submission 1 was something like 45, 15, 368
layer.sizes = c(50, 15, ncol(route.presence.absence))

# Coefficients seemed to have variance of about .02 in a recent test run.
prior.var = c(.01, .01, .01)

# Model blows up with learning.rate == 0.1
learning.rate = .01

# Trial and error in submission 1 indicated that more importance samples was 
# better.
n.importance.samples = 30
n.minibatch = 10

net = mistnet(
  x = env[in.train, ],
  y = route.presence.absence[in.train, ],
  layer.definitions = list(
    defineLayer(
      nonlinearity = rectify.nonlinearity(),
      size = layer.sizes[1],
      prior = gaussianPrior(0, prior.var[1])
    ),
    defineLayer(
      nonlinearity = linear.nonlinearity(),
      size = layer.sizes[2],
      prior = gaussianPrior(0, prior.var[2])
    ),
    defineLayer(
      nonlinearity = sigmoid.nonlinearity(),
      size = layer.sizes[3],
      prior = gaussianPrior(0, prior.var[3])
    )
  ),
  loss = bernoulliRegLoss(a = 1 + 1E-7),
  updater = adagrad.updater(learning.rate = learning.rate),
  sampler = gaussianSampler(ncol = 10, sd = 1),
  n.importance.samples = n.importance.samples,
  n.minibatch = n.minibatch,
  training.iterations = 0
)

# Currently, mistnet does not initialize the coefficients automatically.
# This gets it started with nonzero values.
for(layer in net$layers){
  layer$coefficients[ , ] = rt(length(layer$coefficients), df = 5) * .1
  
  # Biases can move more freely than coefficients
  layer$bias.updater$learning.rate = layer$bias.updater$learning.rate * 10
}

# Initialize the biases of the final layer
net$layers[[3]]$biases[] = qlogis(colMeans(route.presence.absence[in.train, ]))

system.time({
  for(i in 1:200){
    net$fit(100)
    assert_that(!any(is.nan(net$layers[[3]]$outputs)))
    # Update prior mean of last layer.  Pull it in sligthly from the
    # observed mean, as if there were one observation at exactly 0.
    environment(net$layers[[3]]$prior$getLogGrad)$mean = rowMeans(
      net$layers[[3]]$coefficients
    ) * ncol(net$y) / (ncol(net$y) + 1)
    cat(".")
  }
})

lattice::levelplot(net$layers[[1]]$coefficients)
plot(prcomp(net$layers[[1]]$coefficients[-(1:8), ]), npcs = 10)
