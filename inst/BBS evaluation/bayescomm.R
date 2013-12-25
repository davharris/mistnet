load("birds.Rdata")
library(devtools)

# Install the same version of BayesComm that I used for the analysis
install_github(
  username = "davharris",
  repo = "BayesComm",
  ref = "63fc30773cf57f8c6411789da58ffbd3439b3e62"  
)
library(BayesComm)

set.seed(1)

env = x[ , grep("^bio", colnames(x))]

# Even with the memory optimizations I added, I can only fit about 500
# iterations in memory.
# I can probably get about 35000 iterations in with 15 hours of computation,
# so let's thin by 60 and burn in for 5000 iterations to leave 500 samples.
system.time({
  bc.model = BC(
    Y = route.presence.absence[in.train, ],
    X = env[in.train, ],
    model = "full",
    its  = 35000,
    thin = 60,
    burn = 5000
  )
  
  bc.predictions = BayesComm:::predict.bayescomm(
    bc.model,
    env[in.test, ]
  )
})
print(object.size(bc.model), units = "Mb")

save(bc.model, file = "bc.model.Rdata")
save(bc.predictions, file = "bc.predictions.Rdata")