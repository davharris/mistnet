devtools::load_all()
load("birds.Rdata")


# Build CV folds ----------------------------------------------------------

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

# priors distributed log-uniformly between .001 and 1
prior.var1 = 10^(runif(1, min = -4, max = 0))
prior.var2 = 10^(runif(1, min = -4, max = 0))
prior.var3 = 10^(runif(1, min = -4, max = 0))

# minibatch.size distributed uniformly between 10 and 100
minibatch.size = sample(10:100, 1)

# n.ranef distributed uniformly between 10 and 50
n.ranef = sample(10:50, 1)

# n.importance.samples distributed uniformly between 10 and 50
n.importance.samples = sample(10:50, 1)

# n.layer2 triangularly distributed between 3 and 25
triangle = function(min, max){
  unscaled = max:min - min + 1
  unscaled / sum(unscaled)
}
n.layer2 = sample(3:25, 1, prob = triangle(3, 25))

# n.layer1 triangluarly distributed between n.layer2 and 50
n.layer1 = sample(n.layer2:50, 1, prob = triangle(n.layer2, 50))

# Starting learning rate currently fixed at 1E-3
starting.rate = 1E-3
