# devtools::install_github("davharris/mistnet")

library(mistnet)
set.seed(1)

load("birds.Rdata")
load("fold.ids.Rdata")

env = as.data.frame(x[ , grep("^bio", colnames(x))])

# How many samples to generate when evaluating CV fit
n.prediction.samples = 250L

# Number of times to do fit & evaluate loop. Total training time is thus up to
# cv.seconds * n.iterations * n.folds, plus prediction time.
n.iterations = 12L

# Random log-uniform samples between min and max
rlunif = function(n, min, max){
  as.integer(floor(exp(runif(n, log(min), log(max + 1)))))
}


# Choose hyperparams ------------------------------------------------------

hyperparams = data.frame(
  n.minibatch = rlunif(n.iterations, 10, 100),
  sampler.size = rlunif(n.iterations, 5, 20),
  n.importance.samples = rlunif(n.iterations, 20, 50),
  n.layer1 = rlunif(n.iterations, 20, 50),
  n.layer2 = rlunif(n.iterations, 5, 20),
  learning.rate = 0.1,
  fit.seconds = 12 * 60
)


# Fitting code ------------------------------------------------------------

fit = function(x, y, hyperparams, i){
  net = mistnet(
    x = x,
    y = y,
    layer.definitions = list(
      defineLayer(
        nonlinearity = rectify.nonlinearity(),
        size = hyperparams$n.layer1[i],
        prior = gaussian.prior(mean = 0, sd = .1)
      ),
      defineLayer(
        nonlinearity = linear.nonlinearity(),
        size = hyperparams$n.layer2[i],
        prior = gaussian.prior(mean = 0, sd = .1)
      ),
      defineLayer(
        nonlinearity = sigmoid.nonlinearity(),
        size = ncol(y),
        prior = gaussian.prior(mean = 0, sd = .1)
      )
    ),
    loss = bernoulliRegLoss(a = 1 + 1E-6),
    updater = adagrad.updater(learning.rate = hyperparams$learning.rate[i]),
    sampler = gaussian.sampler(ncol = hyperparams$sampler.size[i], sd = 1),
    n.importance.samples = hyperparams$n.importance.samples[i],
    n.minibatch = hyperparams$n.minibatch[i],
    training.iterations = 0
  )
  # Currently, mistnet does not initialize the coefficients automatically.
  # This gets it started with nonzero values.
  for(layer in net$layers){
    layer$coefficients[ , ] = rnorm(length(layer$coefficients), sd = .1)
  }
  net$layers[[1]]$biases[] = 1 # First layer biases equal 1
  start.time = Sys.time()
  while(
    difftime(Sys.time(), start.time, units = "secs") < hyperparams$fit.seconds[i]
  ){
    if(is.nan(net$layers[[3]]$outputs[[1]])){
      stop("NaNs detected :-(")
    }
    net$fit(10)
    cat(".")
    # Update prior variance
    for(layer in net$layers){
      layer$prior$update(
        layer$coefficients, 
        update.mean = FALSE, 
        update.sd = TRUE,
        min.sd = .01
      )
    }
    # Update mean for final layer
    net$layers[[3]]$prior$update(
      layer$coefficients, 
      update.mean = TRUE, 
      update.sd = FALSE,
      min.sd = .01
    )
  } # End while
  
  net
}


# Cross-validation --------------------------------------------------------

out = list()

for(i in 1:n.iterations){
  cat(paste0("Starting iteration ", i, "\n"))
  for(fold.id in 1:max(fold.ids)){
    cat(paste0(" Starting fold ", fold.id, "\n  "))
    in.fold = fold.ids != fold.id
    net = fit(
      scale(env)[in.train, ][in.fold, ], 
      y = route.presence.absence[in.train, ][in.fold, ],
      hyperparams = hyperparams,
      i = i
    )
    
    cat("\n evaluating")
    
    loglik = importanceSamplingEvaluation(
      net, 
      newdata = scale(env)[in.train, ][!in.fold, ],
      y = route.presence.absence[in.train, ][!in.fold, ],
      batches = 10L,
      batch.size = n.prediction.samples / 10L,
      verbose = TRUE
    )
    cat("\n")
    out[[length(out) + 1]] = c(
      iteration = i, 
      fold = fold.id, 
      seconds = hyperparams$fit.seconds[i],
      loglik = mean(loglik)
    )
    
  } # End fold
} # End iteration


# Save CV results ---------------------------------------------------------

mistnet.results = merge(
  x = as.data.frame(do.call(rbind, out)),
  y = cbind(iteration = 1:nrow(hyperparams), hyperparams)
)

save(mistnet.results, file = "mistnet-results.Rdata")


# fit final model ---------------------------------------------------------

library(dplyr)

logliks = mistnet.results %>% group_by(iteration) %>% summarize(mean(loglik))

net = fit(
  x = scale(env)[in.train, ], 
  y = route.presence.absence[in.train, ], 
  hyperparams, 
  which.max(logliks[,2])
)

save(net, file = "mistnet.model.Rdata")
