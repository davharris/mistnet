context("Backprop")
set.seed(1)

test_that("3-layer backprop works", {
  # Need a test with three layers because the code treats the first and last
  # layers differently.  Need to make sure the code still works with the
  # intermediate layers that are neither first nor last.
  eps = 1E-5
  
  x = matrix(rnorm(1819), ncol = 17, nrow = 107)
  y = dropoutMask(107, 14)
  
  net = mistnet(
    x = x,
    y = y,
    nonlinearity.names = c("rectify", "rectify", "sigmoid"),
    hidden.dims = c(13L, 17L),
    priors = list(
      gaussian.prior(mean = 0, var = .001),
      gaussian.prior(mean = 0, var = .001),
      gaussian.prior(mean = 0, var = .001)
    ),
    loss = crossEntropy,
    lossGrad = crossEntropyGrad,
    minibatch.size = 13L,
    n.importance.samples = 1L,
    n.ranef = 7L,
    ranefSample = gaussianRanefSample,
    training.iterations = 0L
  )
  
  net$fit(1)
  # If all the coefficients in layer 2 are 0, then the
  # weights in layer 1 can't possibly matter
  expect_true(all(net$layers[[1]]$weighted.llik.grads == 0))
  
  
  
  net$layers[[1]]$coefficients[ , ] = rnorm(length(net$layers[[1]]$coefficients[ , ])) / 1000
  net$layers[[2]]$coefficients[ , ] = rnorm(length(net$layers[[2]]$coefficients[ , ]))
  net$layers[[3]]$coefficients[ , ] = rnorm(length(net$layers[[3]]$coefficients[ , ]))
  
  net$selectMinibatch()
  set.seed(1)
  # feedforward, backprop, average sample gradients. Don't update.
  net$estimateGradient() 
  grad = net$layers[[1]]$weighted.llik.grads[1,1]
  
  net$layers[[1]]$coefficients[1, 1] = net$layers[[1]]$coefficients[1, 1] + eps
  set.seed(1)
  net$estimateGradient()
  plus.loss = mean(
    rowSums(
      net$loss(
        y = net$y[net$minibatch.ids, ], 
        yhat = net$layers[[3]]$outputs[,,1]
      )
    )
  )
  
  # 2*eps: once to undo the plus above, once to actually decrement
  net$layers[[1]]$coefficients[1, 1] = net$layers[[1]]$coefficients[1, 1] - 2 * eps
  set.seed(1)
  net$estimateGradient()
  minus.loss = mean(
    rowSums(
      net$loss(
        y = net$y[net$minibatch.ids, ], 
        yhat = net$layers[[3]]$outputs[,,1]
      )
    )
  )
  expect_equal(
    grad,
    (plus.loss - minus.loss)/2 /eps,
    tolerance = 1E-7
  )
})
