library(fastICA)
library(mistnet)
library(GPArotation)


load("birds-traits.Rdata")

# Code below expects prior_means to have sd of 1
stopifnot(all.equal(apply(prior_means, 2, sd), rep(1, ncol(prior_means)), check.attributes = FALSE))

# Add extra columns for anonymous environmental influences
prior_means = cbind(prior_means, matrix(0, nrow = nrow(prior_means), ncol = 5))


set.seed(1)

ic_env = fastICA(scale(env), 6)$S

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
  updater = adagrad.updater(learning.rate = 0.1),
  initialize.weights = FALSE,
  initialize.biases = FALSE,
  n.minibatch = 25,
  n.importance.samples = 35,
  sampler = gaussian.sampler(ncol = ncol(prior_means), sd = 1L)
)

message("priors initializing weights...")

for(i in 1:length(net$layers)){
  layer = net$layers[[i]]
  
  # According to http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
  # ... the paper at arXiv:1502.01852 recommends initializing with variance of 2/n_inputs
  init_var = 2 / layer$coef.dim[1]
  
  # Update prior
  layer$prior$sd = sqrt(init_var)
  
  warning("figure out transposition")
  
  # Orthogonal matrix with the variance proposed above
  w = t(Random.Start(max(layer$coef.dim)))[1:layer$coef.dim[1], 1:layer$coef.dim[2]]
  w = w / sqrt(var(c(w))) * sqrt(init_var)
  
  layer$weights[] = w
}

prior_sd = net$layers[[3]]$prior$sd
scaled_prior_means = apply(prior_means * prior_sd, 2, function(x) x - mean(x))

# Initialize the third layer near its prior, but with mean zero
net$layers[[3]]$weights = t(scaled_prior_means) + net$layers[[3]]$weights

message("Initializing biases...")

#net$layers[[1]]$biases[] = 1
#net$layers[[2]]$biases[] = 1

# Third layer: 
# Start with final-layer initialization based on regularizedcolumn means
net$layers[[3]]$biases[] = qlogis(
  (1 + colSums(net$y)) / (2 + nrow(net$y))
)

bias_grad = 0
for(i in 1:10){
  net$selectMinibatch()
  net$estimateGrad()
  
  # Adjust third layer's biases so that the species means are about right.
  # Because the last layer's weights are non-standard, I'm optimiging this by gradient descent
  # rater than using a rule of thumb.
  bias_grad = net$layers[[3]]$weighted.bias.grads + .1 * bias_grad
  net$layers[[3]]$biases = net$layers[[3]]$biases - 1 / (5 + i) * bias_grad
}





library(progress)
maxit = 500
ticksize = 10
pb <- progress_bar$new(total = maxit)
pb$tick(0)

message("fitting...")


for(i in 1:(maxit / ticksize)){
  net$fit(ticksize)
  show(lattice::levelplot(sqrt(net$layers[[2]]$coef.updater$squared.grad)))
  pb$tick(ticksize)
}

sort(structure(apply(net$layers[[2]]$outputs, 2, sd), names = colnames(prior_means)))

pred = predict(net, ic_env, 100, return.model = TRUE)
structure(round(apply(pred$layers[[2]]$outputs, 2, max), 3), names = colnames(prior_means))

