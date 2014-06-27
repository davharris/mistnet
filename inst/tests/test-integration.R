context("Integration tests")

test_that("one-layer network finds correct parameters",{
  # Random effect doesn't do anything here
  
  set.seed(1)
  
  n.importance.samples = 1L
  n.minibatch = 21L
  learning.rate = .1
  n.spp = 50
  
  x = matrix(rnorm(1E4), ncol = 5)
  
  true.coefs = matrix(rnorm(ncol(x)* n.spp), ncol = n.spp)
  
  # No sampling noise: y is probabilities
  y = matrix(
    sigmoid(x %*% true.coefs),
    ncol = n.spp
  )
  
  net = mistnet(
    x = x,
    y = y,
    layer.definitions = list(
      defineLayer(
        nonlinearity = sigmoid.nonlinearity(), 
        size = ncol(y), 
        prior = gaussianPrior(0, .1)
      )
    ),
    sampler = gaussianSampler(ncol = 1, sd = 1),
    loss = bernoulliLoss(),
    updater = sgd.updater(learning.rate = .01, momentum = .9)
  )
  
  # Give the network a warm start
  net$layers[[1]]$coefficients[-6, ] = true.coefs + rnorm(length(true.coefs), sd = .25)
  
  
  estimated.coefs = net$layers[[1]]$coefficients[-6, ]
  r.squared = 1 - sum((true.coefs - estimated.coefs)^2) / sum((true.coefs - mean(true.coefs))^2)
  
  # Model should fail before fitting
  expect_false(r.squared > .99)
  
  net$fit(100)
  
  estimated.coefs = net$layers[[1]]$coefficients[-6, ]
  
  r.squared = 1 - sum((true.coefs - estimated.coefs)^2) / sum((true.coefs - mean(true.coefs))^2)
  
  expect_true(r.squared > .99)
  
  
  # With a strong prior, the coefficients should shrink toward zero
  environment(net$layers[[1]]$prior$getLogGrad)$var = .0001
  net$fit(1)
  expect_true(
    mean(abs(estimated.coefs) > abs(net$layers[[1]]$coefficients[-6, ])) > .99
  )
})

