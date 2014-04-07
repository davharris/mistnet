set.seed(1)

n.species = 300
n.sites = 1500
n.factors = 5 # Rank of env %*% coefs
iter = 1E2

env = matrix(rnorm(n.sites * n.factors), ncol = n.factors)
coefs = matrix(rnorm(n.species * n.factors, sd = 1/2), nrow = n.factors)
biases = rnorm(n.species, mean = -3, sd = 2)


inputs = (env %*% coefs) %plus% biases

# Take the output of a Laplace distribution and square it.
# This makes the lateral coefficients look like what mistnet finds with
# real data (at least visually).
pre.lateral = sample(
  c(-.005, .005) * rexp(n.species^2)^2
)
dim(pre.lateral) = c(n.species, n.species)
lateral = pre.lateral + t(pre.lateral) # Make symmetric
diag(lateral) = 0



state = sigmoid(inputs)

# Gibbs sampling.  Update one species (column) at a time, conditional 
# on everyone else.
for(i in 1:iter){
  for(j in 1:n.species){
    state[ , j] = rbinom(
      n.sites,
      prob = sigmoid(inputs[, j] + state %*% lateral[, j]),
      size = 1
    )
  }
}


fakedata = state

save(coefs, fakedata, env, lateral, biases = biases, file = "inst/fakedata.Rdata")
