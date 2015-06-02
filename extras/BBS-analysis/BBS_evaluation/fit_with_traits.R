library(fastICA)
library(mistnet)
library(progress)

prior_means = readRDS("bbs_data/prior_means.rds")
env = readRDS("bbs_data/env.rds")
runs = readRDS("bbs_data/runs.rds")
pa = readRDS("bbs_data/pa.rds")

# Code below expects prior_means to have sd of 1
stopifnot(all.equal(apply(prior_means, 2, sd), rep(1, ncol(prior_means)), check.attributes = FALSE))

# Add extra columns for anonymous environmental influences
prior_means = cbind(prior_means, matrix(0, nrow = nrow(prior_means), ncol = 5))

set.seed(12345)

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
  updater = adagrad.updater$new(learning.rate = 0.1, K = .1),
  initialize.weights = TRUE,
  initialize.biases = TRUE,
  n.minibatch = 25,
  n.importance.samples = 35,
  sampler = gaussian.sampler(ncol = 10L, sd = 1)
)

message("initializing priors...")
for(layer in net$layers){
  layer$prior$sd = sd(layer$weights)
}
prior_sd = net$layers[[3]]$prior$sd
scaled_prior_means = apply(prior_means * prior_sd, 2, function(x) x - mean(x))


message("Initializing final layer...")

# Initialize the third layer near a rescaled version of its prior (mean zero, standardized SD)
net$layers[[3]]$weights = t(scaled_prior_means) + net$layers[[3]]$weights



message("fitting...")
maxit = 500
ticksize = 10
pb <- progress_bar$new(total = maxit)
pb$tick(0)

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
  1:20,
  function(i){
    sapply(1:nrow(pa), function(row){sum(w[row, ] * pred$layers[[2]]$outputs[row, i, ])})
  }
)
color = z - min(z)
color = color / max(color)

for(i in 1:20){
  plot(
    runs$lati ~ runs$loni, 
    col = rgb(sqrt(color[,i]),0,0),
    main = colnames(prior_means)[i],
    pch = 16,
    cex = 0.7,
    asp = 1
  )
}

par(mfrow = c(2, 1))
for(i in 1:20){
  hist(
    pred$layers[[2]]$outputs[row, i, ]
  )
  hist(
    sample(
      pred$layers[[2]]$outputs[row, i, ],
      size = 1E3,
      prob = w[i, ],
      replace = TRUE
    ),
    xlim = range(pred$layers[[2]]$outputs[row, i, ])
  )
}