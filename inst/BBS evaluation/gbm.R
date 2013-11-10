load("birds.Rdata")
library(gbm)

set.seed(1)

interaction.depths = c(1, 2, 3, 5, 8)

env = as.data.frame(x[ , grep("^bio", colnames(x))])

species.gbm = function(species.num, interaction.depths){
  # For each interaction depth, find the number of trees that minimizes the 
  # species' CV error.  Return the set of predictions from the model with the 
  # best combination of tree number and tree depth.
  
  
  cat(colnames(route.presence.absence)[[species.num]], "\n")
  predictions = matrix(
    NA, 
    nrow = sum(in.test), 
    ncol = length(interaction.depths)
  )
  errors = numeric(length(interaction.depths))
  
  for(i in seq_along(interaction.depths)){
    
    model = gbm(
      route.presence.absence[in.train, species.num] ~ .,
      distribution = "bernoulli",
      data = env[in.train, ], 
      n.trees = 1E4,
      interaction.depth = interaction.depths[[i]],
      shrinkage = 0.001,
      cv.folds = 5
    )
    
    errors[[i]] = min(model$cv.error)
    
    # Making predictions for each tree depth does eat some extra time, but
    # it means I don't have to hold multiple models in memory.
    predictions[ , i] = predict(
      model, 
      env[in.test, ], 
      gbm.perf(model, plot = FALSE), 
      type = "response"
    )
  }
  
  predictions[ , which.min(errors)]
}

system.time({
  gbm.predictions = sapply(
    seq_len(ncol(route.presence.absence)),
    function(i){
      species.gbm(species.num = i, interaction.depth = interaction.depths)
    }
  )
})

save(gbm.predictions, file = "gbm.predictions.Rdata")