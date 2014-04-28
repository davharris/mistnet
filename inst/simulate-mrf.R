devtools::load_all()
set.seed(1)

n.species = 100
n.sites = 2000
n.factors = 5 # Rank of env %*% coefs
iter = 1E3

env = matrix(rnorm(n.sites * n.factors), ncol = n.factors)
coefs = matrix(rnorm(n.species * n.factors, sd = 1/2), nrow = n.factors)
biases = rnorm(n.species, mean = 0, sd = 1/2)


inputs = (env %*% coefs) %plus% biases

lateral = sample(
  c(-.1, .1) * rexp(n.species^2)
)
dim(lateral) = c(n.species, n.species)
lateral = lateral + t(lateral) # Make symmetric
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
