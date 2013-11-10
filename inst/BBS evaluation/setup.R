devtools::load_all()
load("birds.Rdata")

# Discrete triangular distribution PDF:
# Defined so that small values in the allowed range have the highest frequency,
# like the log-uniform. For some cases (e.g. limits between 10 and 25), 
# log-uniform felt too flat, so I tried this instead.
triangle = function(min, max){
  unscaled = max:min - min + 1
  unscaled / sum(unscaled)
}


# Build CV folds ----------------------------------------------------------
# Setting a RNG seed so that I can ensure that all methods see the same CV 
# splits
set.seed(1)
n.folds = 5L
fold.ids = sample(rep(1:n.folds, length = sum(in.train)))

# Ensure that no folds have completely missing or completely present species
# during CV training.  
range.colmeans = range(
  sapply(
    1:n.folds, 
    function(i) colMeans(route.presence.absence[in.train, ][fold.ids != i, ])
  )
)
stopifnot(min(range.colmeans) > 0, max(range.colmeans) < 1)


# Select hyperparameters --------------------------------------------------

# hyperparameter selection should be random; not based on the seed used above
# for selecting CV folds
set.seed(NULL)

# priors distributed log-uniformly between .001 and 1
prior.var1 = 10^(runif(1, min = -4, max = 0))
prior.var2 = 10^(runif(1, min = -4, max = 0))
prior.var3 = 10^(runif(1, min = -4, max = 0))

# minibatch.size distributed triangularly between 10 and 100
# Smaller minibatches allow for more gradient updates, bigger ones are more
# precise.
# I think the 10-100 range comes from Hinton's "Practical Guide" to RBMs.
minibatch.size = sample(10:100, 1, prob = triangle(10, 100))

# n.ranef distributed uniformly between 10 and 50.
# The "true" number of important dimensions seems to be much smaller than this,
# but with limited sampling effort per iteration and Gaussian priors, more
# seemed to work better.
n.ranef = sample(10:50, 1)

# n.importance.samples distributed uniformly between 10 and 50.
# Tang and Salakhutdinov seemed to find the best performance with 20-30 samples.
n.importance.samples = sample(10:50, 1)

# n.layer2 triangularly distributed between 5 and 25.
# This layer does dimensionality reduction. With better priors, fewer dimensions
# would probably be needed, since there still tend to be strong correlations.
n.layer2 = sample(5:25, 1, prob = triangle(5, 25))

# n.layer1 uniformly distributed between n.layer2 and 50
# This layer does basis expansion, so it should be at least as wide as layer2
n.layer1 = sample(n.layer2:50, 1)

# Starting learning rate currently fixed at 1E-3.
# Found this through trial and error.  Haven't messed with it much, since I 
# don't want any exploding gradients and I already have two other ways to
# trade off speed versus accuracy during gradient descent.
starting.rate = 1E-3
