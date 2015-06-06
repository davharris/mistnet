library(mistnet)
library(progress)
`%plus%` = mistnet:::`%plus%`


set.seed(1)

n.species = 120
n.sites = 2000
n.factors = 5 # Rank of env %*% coefs
iter = 1E3

env = matrix(rnorm(n.sites * n.factors), ncol = n.factors)
coefs = matrix(rnorm(n.species * n.factors, sd = 1/2), nrow = n.factors)
biases = rnorm(n.species, mean = 1, sd = 1/2)


inputs = (env %*% coefs) %plus% biases

lateral = matrix(0, n.species, n.species)
lateral[upper.tri(lateral)] = sample(
  c(-1, -1, -1, 1) * abs(rt(choose(n.species, 2), df = 10)) / 5
)
lateral = lateral + t(lateral) # Make symmetric



state = mistnet:::sigmoid(inputs)

pb <- progress_bar$new(
  format = "  simulating landscape [:bar] :percent eta: :eta",
  total = iter,
  clear = FALSE
)

# Gibbs sampling.  Update one species (column) at a time, conditional 
# on everyone else.
for(i in 1:iter){
  for(j in sample(1:n.species)){
    # Conditional probability of occurrence for species j
    prob = mistnet:::sigmoid(inputs[, j] + state %*% lateral[, j])
    
    # Random sample from conditional distribution of species j
    state[ , j] = rbinom(n.sites, prob = prob, size = 1)
  }
  pb$tick()
}


fakedata = state

save(coefs, fakedata, env, lateral, biases = biases, file = "extras/mrf/fakedata.Rdata")
