library(fastICA)
library(mistnet)
library(progress)

nonzero_prior_means = readRDS("bbs_data/prior_means.rds")
env = readRDS("bbs_data/env.rds")
runs = readRDS("bbs_data/runs.rds")
pa = readRDS("bbs_data/pa.rds")
species =readRDS("bbs_data/species.rds") 

# Proportion of variance in pre-initialized weights that should be explained
# by the prior mean.
prior_proportion = 0.95

# Add extra columns for anonymous environmental influences
prior_means = cbind(nonzero_prior_means, matrix(0, nrow = nrow(nonzero_prior_means), ncol = 5))

set.seed(98374)

ic_env = fastICA(scale(env), 6)$S

# Create the network object. Note that priors are (numeric) NAs for now.
net = mistnet(
  x = ic_env[runs$in_train, ],
  y = pa[runs$in_train, ],
  layer.definitions = list(
    defineLayer(
      nonlinearity = leaky.rectify.nonlinearity(),
      size = 50,
      prior = gaussian.prior(mean = 0, sd = NA * 1)
    ),
    defineLayer(
      nonlinearity = leaky.rectify.nonlinearity(),
      size = ncol(prior_means),
      prior = gaussian.prior(mean = 0, sd = NA * 1)
    ),
    defineLayer(
      nonlinearity = sigmoid.nonlinearity(),
      size = ncol(pa),
      prior = gaussian.prior(mean = t(prior_means), sd = NA * 1)
    )
  ),
  loss = bernoulliRegLoss(1 + 1E-5, 1 + 1E-5),
  updater = rmsprop.updater$new(learning.rate = 0.01, decay = 0.1),
  initialize.weights = TRUE,
  initialize.biases = TRUE,
  n.minibatch = 25,
  n.importance.samples = 35,
  sampler = gaussian.sampler(ncol = 5L, sd = 1)
)

message("initializing priors...")
for(layer in net$layers){
  layer$prior$sd = sd(layer$weights)
}

# Target standard deviations.  For anonymous variables, use the full amount.
# For the others, cut the variance by a factor of 1-prior_proportion (sqrt for SD).
is_anonymous = grepl("\\d", colnames(prior_means))
prior_sds = net$layers[[3]]$prior$sd * ifelse(is_anonymous, 1, sqrt(1 - prior_proportion))
net$layers[[3]]$prior$sd  = prior_sds


# Target means, with specified SD
net$layers[[3]]$prior$mean = t(
  sapply(
    1:ncol(prior_means),
    function(i){
      if(sd(prior_means[ , i]) == 0){
        rep(0, nrow(prior_means))
      }else{
        prior_means[ , i] / sd(prior_means[ , i]) * max(prior_sds) * sqrt(prior_proportion)
      }
    }
  )
)

message("Initializing final layer...")

# Initialize the third layer near scaled_prior_means.
# Divide the initialized weights by sqrt(2) to reduce their variance to prior_proportion
net$layers[[3]]$weights = net$layers[[3]]$prior$mean * sqrt(prior_proportion) + 
  net$layers[[3]]$weights * ifelse(is_anonymous, 1, sqrt(1 - prior_proportion))


net$layers[[3]]$coef.updater$learning.rate = net$layers[[3]]$coef.updater$learning.rate

for(i in 1:5){
  net$selectMinibatch()
  net$estimateGrad()
  layer = net$layers[[3]]
  layer$bias.updater$computeDelta(layer$weighted.bias.grads/net$row.selector$n.minibatch)
  layer$biases = layer$biases + layer$bias.updater$delta
}

message("fitting...")
maxit = 500
ticksize = 10
pb <- progress_bar$new(total = maxit)

for(i in 1:(maxit / ticksize)){
  net$fit(ticksize)
  show(lattice::levelplot(sqrt(net$layers[[2]]$coef.updater$squared.grad)))
  pb$tick(ticksize)
}

sort(structure(apply(net$layers[[2]]$outputs, 2, sd), names = colnames(prior_means)))
sort(apply(net$layers[[3]]$weights - net$layers[[3]]$prior$mean, 1, sd))


pred = predict(net, ic_env, 100, return.model = TRUE)
sort(structure(round(apply(pred$layers[[2]]$outputs, 2, sd), 3), names = colnames(prior_means)))

loglik = apply(
  pred$layers[[3]]$outputs, 
  3, 
  function(x){
    -rowSums(bernoulliLoss()$loss(y = pa, yhat = x))
  }
)

w = t(apply(loglik, 1, function(x) x / sum(x)))

z = sapply(
  1:10,
  function(i){
    sapply(1:nrow(pa), function(row){sum(w[row, ] * pred$layers[[2]]$outputs[row, i, ])})
  }
)
color = z - min(z)
color = apply(color, 2, function(x)x/max(x))

color2 = apply(pred$layers[[2]]$outputs, 2, rowMeans)
color2 = color2 - min(color2)
color2 = apply(color2, 2, function(x)x/max(x))

color3 = pmax(z - apply(pred$layers[[2]]$outputs, 2, rowMeans), 0)
color2 = apply(color3, 2, function(x)x/max(x))

for(i in 1:10){
  cols = topo.colors(100)
  
  plot(
    runs$lati ~ runs$loni, 
    col = cols[ceiling(color[,i] * 100)],
    main = paste0(i, ": ", colnames(prior_means)[i]),
    pch = 16,
    cex = 0.7,
    asp = 1
  )
}
