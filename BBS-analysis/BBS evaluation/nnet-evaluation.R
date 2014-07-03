load("birds.Rdata")
library(nnet)
load("fold.ids.Rdata")

set.seed(1)

env = as.data.frame(x[ , grep("^bio", colnames(x))])

results = data.frame(
  log.likelihood = numeric(0), 
  size = numeric(0), 
  decay = numeric(0),
  fold.id = numeric(0), 
  i = numeric(0)
)

i = 0
start.time = Sys.time()

# allocate 14+ hours for hyperparameter optimization
while(
  as.numeric(Sys.time()) - as.numeric(start.time) < 14 * 60 * 60
){
  i = i + 1
  cat(i, "\n")
  # hyperparameters:
  decay = rexp(1, rate = 1)
  size = floor(runif(1, min = 1, max = 51)) # integer between 1 and 50
  
  
  for(fold.id in 1:max(fold.ids)){
    in.fold = fold.ids != fold.id
    
    net = nnet(
      x = scale(env)[in.train, ][in.fold, ],
      y = route.presence.absence[in.train, ][in.fold, ],
      maxit = 1000,
      MaxNWts = 20000,
      decay = decay,
      size = size,
      entropy = TRUE
    )
    
    
    log.likelihood = sum(
      dbinom(
        route.presence.absence[in.train, ][!in.fold, ],
        size = 1, 
        prob = predict(net, scale(env)[in.train, ][!in.fold, ]),
        log = TRUE
      )
    )
    results = rbind(
      results, 
      data.frame(
        log.likelihood = log.likelihood, 
        size = size, 
        decay = decay,
        fold.id = fold.id,
        i = i
      )
    )
  }
}

save(results, file = "nnet.cv.results.Rdata")

library(plyr)

compiled.results = ddply(
  results, 
  c("i", "size", "decay"), 
  function(x) c(log.likelihood = mean(x$log.likelihood))
)

optimal.size = compiled.results$size[which.max(compiled.results$log.likelihood)]
optimal.decay = compiled.results$decay[which.max(compiled.results$log.likelihood)]


net = nnet(
  x = scale(env)[in.train, ],
  y = route.presence.absence[in.train, ],
  maxit = 1000,
  MaxNWts = 20000,
  decay = optimal.decay,
  size = optimal.size,
  entropy = TRUE
)

nnet.predictions = predict(net, scale(env)[in.test, ])

save(nnet.predictions, file = "nnet.predictions.Rdata")
Sys.time() - start.time
