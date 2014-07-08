load("birds.Rdata")
env = scale(x[, 1:8])
devtools::load_all()

o = sample.int(sum(in.train))

# Best model in first submission's cross-validation was something like 45, 15
layer.sizes = c(30, 12, ncol(route.presence.absence))

# Model still runs with learning.rate = 0.1, so maybe that's good?
learning.rate = .1

# Trial and error in submission 1 indicated that more importance samples was 
# better.
n.importance.samples = 30
n.minibatch = 30

n.ranef = 5

net = mistnet(
  x = env[in.train, ][o, ],
  y = route.presence.absence[in.train, ][o, ],
  layer.definitions = list(
    defineLayer(
      nonlinearity = rectify.nonlinearity(),
      size = layer.sizes[1],
      prior = gaussian.prior(mean = 0, var = 1)
    ),
    defineLayer(
      nonlinearity = linear.nonlinearity(),
      size = layer.sizes[2],
      prior = gaussian.prior(mean = 0, var = 1)
    ),
    defineLayer(
      nonlinearity = sigmoid.nonlinearity(),
      size = layer.sizes[3],
      prior = gaussian.prior(mean = 0, var = 1)
    )
  ),
  loss = bernoulliRegLoss(a = 1 + 1E-6),
  updater = adagrad.updater(learning.rate = learning.rate),
  sampler = gaussian.sampler(ncol = n.ranef, sd = 1),
  n.importance.samples = n.importance.samples,
  n.minibatch = n.minibatch,
  training.iterations = 0
)

# Currently, mistnet does not initialize the coefficients automatically.
# This gets it started with nonzero values.
for(layer in net$layers){
  layer$coefficients[ , ] = rnorm(length(layer$coefficients), sd = .1)
  
  # Biases can move more freely than coefficients
  layer$bias.updater$learning.rate = layer$bias.updater$learning.rate * 10
}

# Set a bunch of the first-layer coefficients using the MARS initialization
init = initializeMars(
  x = env[in.train, ],
  y = route.presence.absence[in.train, ]
)
net$layers[[1]]$biases[1:init$n] = init$biases
net$layers[[1]]$coefficients[1:ncol(env), 1:init$n] = init$coefficients



start.params = net$layers[[1]]$coefficients

net$layers[[1]]$biases[] = 1 # First layer biases equal 1

# Initialize the biases of the final layer
net$layers[[net$n.layers]]$biases[] = qlogis(colMeans(route.presence.absence[in.train, ]))

start.time = Sys.time()
while(difftime(Sys.time(), start.time, units = "secs") < 500){
  net$fit(20)
  cat(".")
  # Update prior variance
  for(layer in net$layers){
    layer$prior$update(
      layer$coefficients, 
      update.mean = FALSE, 
      update.var = TRUE,
      min.var = .001
    )
  }
}

lattice::levelplot(net$layers[[1]]$coefficients)
plot(prcomp(net$layers[[1]]$coefficients[-(1:8), ]), npcs = 10)
plot(prcomp(net$layers[[1]]$coefficients[1:8, ]), npcs = 10)
hist(net$layers[[3]]$inputs)
plot(net$layers[[1]]$coefficients[1:8, ] ~start.params[1:8, ], asp = 1);abline(0,1)
