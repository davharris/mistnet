load("birds.Rdata")

devtools::install_github("goldingn/BayesComm", ref = "0d710cda46a6e7427a560ee5b816c8ef5cd03eeb")
library(BayesComm)
library(Rcpp)

set.seed(1)

env = x[ , grep("^bio", colnames(x))]

# Even with the memory optimizations I added, I can only fit about 500
# iterations in memory.
system.time({
  bc.model = BC(
    Y = route.presence.absence[in.train, ],
    X = env[in.train, ],
    model = "full",
    its  = 42000,
    thin = 80,
    burn = 2000,
    verbose = 2
  )
  
  bc.predictions = BayesComm:::predict.bayescomm(
    bc.model,
    env[in.test, ]
  )
})
print(object.size(bc.model), units = "Mb")

save(bc.model, file = "bc.model.Rdata")
save(bc.predictions, file = "bc.predictions.Rdata")
