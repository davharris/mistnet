context("3-layer backprop")
set.seed(1)

test_that("3-layer backprop works", {
  eps = 1E-5
  
  x = matrix(rnorm(1819), ncol = 17, nrow = 107)
  y = dropoutMask(107, 14)
  
  net = network$new(
    x = x,
    y = y,
    layers = list(
      createLayer(
        dim = c(ncol(x), 13L),
        learning.rate = .01,
        momentum = 0.5,
        prior = gaussian.prior(mean = 0, var = .01),
        dataset.size = nrow(x),
        nonlinearity.name = "sigmoid",
        dropout = FALSE
      ),
      createLayer(
        dim = c(13L, 9L),
        learning.rate = .01,
        momentum = 0.5,
        prior = gaussian.prior(mean = 0, var = .25),
        dataset.size = nrow(x),
        nonlinearity.name = "sigmoid"
      ),
      createLayer(
        dim = c(9L, ncol(y)),
        learning.rate = .00001,
        momentum = 0.5,
        prior = gaussian.prior(mean = 0, var = .25),
        dataset.size = nrow(x),
        nonlinearity.name = "sigmoid"
      )
    ),
    loss = crossEntropy,
    lossGradient = crossEntropyGrad,
    minibatch.size = 89L,
    n.layers = 3L
  )
  
  net$layers[[1]]$coefficients[ , ] = rnorm(length(net$layers[[1]]$coefficients[ , ])) / 1000
  net$layers[[2]]$coefficients[ , ] = rnorm(length(net$layers[[2]]$coefficients[ , ]))
  net$layers[[3]]$coefficients[ , ] = rnorm(length(net$layers[[3]]$coefficients[ , ]))
  
  net$newMinibatch()
  net$feedForward()
  
  expect_equal(
    net$layers[[1]]$output,
    net$layers[[2]]$input
  )
  expect_equal(
    net$layers[[2]]$output,
    net$layers[[3]]$input
  )
  net.out = net$layers[[2]]$output  # output from layer 2
  net$layers[[2]]$forwardPass(net$layers[[1]]$output) # correct output given layer 1
  expect_equal(
    net.out,
    net$layers[[2]]$output
  )
  net.out = net$layers[[3]]$output  # output from network
  net$layers[[3]]$forwardPass(net$layers[[2]]$output) # correct output given layer 2
  expect_equal(
    net.out,
    net$layers[[3]]$output
  )
  
  
  
  net$backprop()
  
  
  grad = net$layers[[1]]$llik.grad[1,1]
  
  net$layers[[1]]$coefficients[1, 1] = net$layers[[1]]$coefficients[1, 1] + eps
  net$feedForward()
  plus.loss = sum(net$reportLoss())
  
  # 2*eps: once to undo the plus above, once to actually decrement
  net$layers[[1]]$coefficients[1, 1] = net$layers[[1]]$coefficients[1, 1] - 2 * eps
  net$feedForward()
  minus.loss = sum(net$reportLoss())
  
  expect_equal(
    grad,
    (plus.loss - minus.loss)/2 /eps,
    tolerance = 1E-7
  )
})