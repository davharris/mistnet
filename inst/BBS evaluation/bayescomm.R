load("birds.Rdata")
library(devtools)

# Install the same version of BayesComm that I used for the analysis
install_github(
  username = "goldingn",
  repo = "BayesComm",
  ref = "35b0288ae7839b5976fcd453e3fd7f7b5b86d855"  
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

save(bc.predictions, file = "bc.predictions.Rdata")