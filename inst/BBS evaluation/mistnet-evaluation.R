load("birds.Rdata")
devtools::load_all()
load("fold.ids.Rdata")
library(plyr)

set.seed(1)

num.attempts = 15L

total.time.start = Sys.time()

for(attempt.num in 1:num.attempts){
  message("Evaluating parameter set #", attempt.num)
  
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

# Make sure the code below gets the optimal parameters, not whatever is floating
# in the global environment
rm(list = names(optimal.row))

net = with(
  as.list(results[optimal.row, ]),{
    start.time = Sys.time()
    
    # direct buildNet to look in the local environment for parameters
    localBuildNet = buildNet
    
    net = localBuildNet(
      x = scale(env)[in.train, ],
      y = route.presence.absence[in.train, ]
    )
    while(
      as.double(Sys.time() - start.time, units = "secs") < seconds
    ){
      net$update_all(10L)
    }
    net
  }
)  

mistnet.prediction.array = predict(
  net, 
  scale(env)[in.test, ],
  n.importance.samples = as.integer(1E3)
)


print(Sys.time() - total.time.start)


save(mistnet.prediction.array, file = "mistnet.predictions.Rdata")
