num.attempts = 15L
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
  save(
    cbind(hyperparameters, output.df), 
    file = paste0("mistnet performance ", attempt.num, ".Rdata")
  )
}