context("Prediction")

test_that("Prediction works",{
  
  x = dropoutMask(17L, 37L)
  y = dropoutMask(17L, 19L)
  n.importance.samples = 11L # Number for testing, not for training
  
  
  net = mistnet(
    x,
    y,
    nonlinearity.names = c("sigmoid", "rectify", "sigmoid"),
    hidden.dims = c(5L, 7L),
    priors = list(
      gaussian.prior(mean = 0, var = 1),
      gaussian.prior(mean = 0, var = 1),
      gaussian.prior(mean = 0, var = 1)
    ),
    n.ranef = 3L,
    ranefSample = gaussianRanefSample,
    n.importance.samples = 10L,
    minibatch.size = 10L,
    training.iterations = 0L,
    loss = crossEntropy,
    lossGrad = crossEntropyGrad
  )
  
  p = predict(net, x, n.importance.samples = n.importance.samples)
  
  expect_equal(dim(p), c(dim(y), n.importance.samples))
  expect_true(all(p == 0.5)) # With no coefficients, everything should be 0.5
})