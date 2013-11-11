load("birds.Rdata")
devtools::load_all()
load("fold.ids.Rdata")
library(plyr)

num.attempts = 15L

total.time.start = Sys.time()

for(attempt.num in 1:num.attempts){
  source("inst/BBS evaluation/setup.R")
  
  hyperparameters = list(
    prior.var1 = prior.var1,
    prior.var2 = prior.var2,
    prior.var3 = prior.var3,
    minibatch.size = minibatch.size,
    n.ranef = n.ranef, 
    n.importance.samples = n.importance.samples,
    n.layer2 = n.layer2,
    n.layer1 = n.layer1,
    starting.rate = starting.rate
  )
  
  out = cbind(hyperparameters, attempt.num = attempt.num, output.df)
  source("inst/example.R")
  save(
    out, 
    file = paste0("mistnet performance ", attempt.num, ".Rdata")
  )
}

results = do.call(
  rbind,
  lapply(
    dir(pattern = "mistnet performance.*\\.Rdata"),
    function(x){load(x);out}
  )
)

means = ddply(results, c("seconds", "attempt.num"), function(x) mean(x$loglik))

optimal.seconds = means$seconds[which.max(means$V1)]
optimal.attempt.num = means$attempt.num[which.max(means$V1)]

optimal.row = which(
  optimal.attempt.num == results$attempt.num & 
    optimal.seconds == results$seconds & 
    results$fold.id == 1
)

results[optimal.row, ]

print(Sys.time() - total.time.start)