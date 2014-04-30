context("integration test: one layer")

test_that("one-layer network finds correct parameters",{
  
  set.seed(1)
  
  n.importance.samples = 1L
  minibatch.size = 21L
  n.ranef = 1L
  learning.rate = .1
  n.spp = 50
  
  x = matrix(rnorm(1E4), ncol = 5)
  
  true.coefs = matrix(rnorm(ncol(x)* n.spp), ncol = n.spp)
  
  y = matrix(
    rbinom(length(x %*% true.coefs), size = 1, prob = sigmoid(x %*% true.coefs)),
    ncol = n.spp
  )
  
  net = mistnet(
    x = x,
    y = y,
    hidden.dims = NULL,
    n.ranef = 1,
    nonlinearity.names = "sigmoid",
    loss.name = "crossEntropy",
    updater.arguments = list(learning.rate = .01, momentum = .9)
  )
  
  # Give the network a warm start
  net$layers[[1]]$coefficients[-6, ] = true.coefs * .9
  
  net$fit(100)
  
  estimated.coefs = net$layers[[1]]$coefficients[-6, ]
  
  r.squared = 1 - var(c(true.coefs - estimated.coefs)) / var(c(true.coefs))
  
  expect_true(r.squared > .99)
})