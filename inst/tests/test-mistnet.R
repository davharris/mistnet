context("Mistnet function")

x = dropoutMask(17L, 37L)
y = dropoutMask(17L, 19L)

test_that("mistnet(training.iterations = 0) works",{
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
    learning.rate = 1E-4,
    momentum = .8,
    n.ranef = 3L,
    ranefSample = gaussianRanefSample,
    n.importance.samples = 10L,
    minibatch.size = 10L,
    training.iterations = 0L,
    loss = crossEntropy,
    lossGrad = crossEntropyGrad
  )
  
  expect_equal(net$completed.iterations, 0)
})