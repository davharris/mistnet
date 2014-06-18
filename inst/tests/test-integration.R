context("Integration tests")

test_that("one-layer network finds correct parameters",{
  # Random effect doesn't do anything here
  
  set.seed(1)
  
  n.importance.samples = 1L
  minibatch.size = 21L
  n.ranef = 1L
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
        prior = gaussianPrior(0, 001)
      )
    ),
    n.ranef = 1,
    loss = bernoulliLoss(),
    updater.arguments = list(learning.rate = .01, momentum = .9)
  )
  
  # Give the network a warm start
  net$layers[[1]]$coefficients[-6, ] = true.coefs * .85
  
  net$fit(100)
  
  estimated.coefs = net$layers[[1]]$coefficients[-6, ]
  
  r.squared = 1 - sum((true.coefs - estimated.coefs)^2) / sum((true.coefs - mean(true.coefs))^2)
  
  expect_true(r.squared > .99)
})

