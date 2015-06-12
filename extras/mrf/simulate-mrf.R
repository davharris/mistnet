library(mistnet)
library(progress)
library(beepr)
`%plus%` = mistnet:::`%plus%`


set.seed(1)

n.species = 100
n.sites = 2500
n.factors = 3 # not including squared terms or cross-products
iter = 1E3

# Environment gets linear and squared 
env = matrix(rnorm(n.sites * n.factors), ncol = n.factors)
colnames(env) = c("a", "b", "c")
env = scale(
    model.matrix(
      ~ poly(a, b, c, degree = 2),
      as.data.frame(env)
    )[, -1]
)

library(magrittr)

has_squared_term = colnames(env) %>% 
  gsub("^.*\\)", "", .) %>% 
  grepl("2", .)

coefs = matrix(rnorm(n.species * n.factors * 3, sd = 1/2), nrow = n.factors * 3)
coefs[has_squared_term, ] = -abs(coefs[has_squared_term, ]) * sqrt(2)

biases = rnorm(n.species, mean = 1, sd = 1)


inputs = (env %*% coefs) %plus% biases

lateral = matrix(0, n.species, n.species)
lateral[upper.tri(lateral)] = sample(
  c(-1, -1, 0, 0, 1) * abs(rt(choose(n.species, 2), df = 5)) / 5
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

save(
  coefs, 
  fakedata, 
  env, 
  lateral, 
  biases, 
  file = "extras/mrf/fakedata.Rdata"
)

beep()
