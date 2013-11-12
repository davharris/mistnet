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
  
  source("inst/example.R")
  out = cbind(hyperparameters, attempt.num = attempt.num, output.df)
  
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

# mistnet doesn't play nice with plyr for some reason: using this instead
means = aggregate(
  results$loglik, 
  by = list(
    time = as.factor(results$seconds), 
    attempt = as.factor(results$attempt.num)
  ), 
  FUN = mean
)


# as.character(as.numeric(factor)) avoids converting factors to their index
# and instead treats the level names as numeric
optimal.seconds = as.numeric(as.character(means$time[which.max(means$x)]))
optimal.attempt.num = as.numeric(
  as.character(means$attempt[which.max(means$x)])
)

# deal with stupid floating point errors introduced by conversion to and 
# from factor
approx.equal = function(x, y, eps = 1E-6){
  abs(x - y) < eps
}

optimal.row = which(
  optimal.attempt.num == results$attempt.num & 
    approx.equal(optimal.seconds, results$seconds) & 
    results$fold.id == 1
)

results[optimal.row, ]

print(Sys.time() - total.time.start)
