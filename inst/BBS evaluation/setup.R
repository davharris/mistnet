devtools::load_all()
load("birds.Rdata")

# Triangular distribution PDF
triangle = function(min, max){
  unscaled = max:min - min + 1
  unscaled / sum(unscaled)
}

# Select hyperparameters --------------------------------------------------

# priors distributed log-uniformly between .0001 and 1
prior.var1 = 10^(runif(1, min = -4, max = 0))
prior.var2 = 10^(runif(1, min = -4, max = 0))
prior.var3 = 10^(runif(1, min = -4, max = 0))

# minibatch.size distributed triangularly between 10 and 100
minibatch.size = sample(10:100, 1, prob = triangle(10, 100))

# n.ranef distributed uniformly between 10 and 50
n.ranef = sample(10:50, 1)

# n.importance.samples distributed uniformly between 10 and 50
n.importance.samples = sample(10:50, 1)

# n.layer2 triangularly distributed between 5 and 25
n.layer2 = sample(5:25, 1, prob = triangle(5, 25))

# n.layer1 uniformly distributed between n.layer2 and 50
n.layer1 = sample(n.layer2:50, 1)

# Starting learning rate currently fixed at 1E-3
starting.rate = 1E-3
