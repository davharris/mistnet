library(fastICA)
library(mistnet)


load("birds-traits.Rdata")

set.seed(1)

ic_env = fastICA(scale(env), 6)$S

prior_means = cbind(prior_means, matrix(0, nrow = nrow(prior_means), ncol = 5))


net = mistnet(
  x = ic_env[runs$in_train, ],
  y = pa[runs$in_train, ],
  layer.definitions = list(
    defineLayer(
      nonlinearity = leaky.rectify.nonlinearity(),
      size = 50,
      prior = gaussian.prior(mean = 0, sd = 0.1)
    ),
    defineLayer(
      nonlinearity = leaky.rectify.nonlinearity(),
      size = ncol(prior_means),
      prior = gaussian.prior(mean = 0, sd = 0.1)
    ),
    defineLayer(
      nonlinearity = sigmoid.nonlinearity(),
      size = ncol(pa),
      prior = gaussian.prior(mean = t(prior_means), sd = 0.5)
    )
  ),
  loss = bernoulliRegLoss(1 + 1E-6, 1 + 1E-6),
  updater = adagrad.updater(learning.rate = .1),
  initialize.weights = FALSE,
  initialize.biases = FALSE,
  n.minibatch = 20,
  n.importance.samples = 30
)

message("initializing weights...")
# Initialize weights with Glorot's method
glorot = function(n_j, n_j_plus_1){
  sqrt(6) / (n_j + n_j_plus_1)
}
glorot_range = sapply(net$layers, function(layer){sqrt(6)/sum(sqrt(dim(layer$weights)))})
for(i in 1:length(net$layers)){
  layer = net$layers[[i]]
  layer$weights[] = runif(length(layer$weights), -glorot_range[[i]], glorot_range[[i]])
}

# Initialize the third layer using its prior
net$layers[[3]]$weights = net$layers[[3]]$weights + net$layers[[3]]$prior$mean

message("Initializing biases...")

# Positive inital biases make relu units less likely to "die"
net$layers[[1]]$biases[] = 1
net$layers[[2]]$biases[] = 1

# Initialize third layer's biases so that the species means are about right
grad = 0
for(i in 1:10){
  net$selectMinibatch()
  net$estimateGrad()
  grad = net$layers[[3]]$weighted.bias.grads + .5 * grad
  net$layers[[3]]$biases = net$layers[[3]]$biases - 1 / (5 + i) * grad
}

message("fitting...")
for(i in 1:50){
  cat(".")
  net$fit(10)
}
