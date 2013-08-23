context("Updating biases")

test_that("biases update correctly",{
  
  eps = 1E-5
  x = matrix(rnorm(1000), nrow = 50)
  y = dropoutMask(50, 17)
  
  net = network(
    x = x,
    y = y,
    layers = list(
      createLayer(
        dim = c(ncol(x), ncol(y)),
        learning.rate = eps,
        momentum = .8,
        prior = gaussian.prior$new(mean = 0, var = 1),
        dataset.size = 12345,
        nonlinearity.name = "sigmoid"
      )
    ),
    n.layers = 1L,
    minibatch.size = 10L,
    loss = crossEntropy,
    lossGradient = crossEntropyGrad
  )
  
  initial.biases = rnorm(ncol(y))
  net$layers[[1]]$biases = initial.biases
  # Currently, gradient isn't directly reported.  Have to infer it from how biases
  # are updated.  Not the most maintainable decision I've ever made...
  net$fit(1)
  
  # factor of 10 is baked into bias update.  See comment there.
  grad = (initial.biases - net$layers[[1]]$biases)
  
  
  net$layers[[1]]$coefficients[ , ] = 0
  
  # With reset coefficients, let's see what the loss is.
  net$feedForward()
  updated.loss = sum(net$reportLoss())
  
  net$layers[[1]]$biases[] = initial.biases
  
  # the updated loss should be less than what was seen with the initial biases
  net$feedForward()
  initial.loss = sum(net$reportLoss())
  expect_true(initial.loss > updated.loss)
  
  bias.to.update = sample.int(length(initial.biases), 1)
  
  net$layers[[1]]$biases[bias.to.update] = initial.biases[bias.to.update] + eps
  net$feedForward()
  plus.loss = sum(net$reportLoss())
  
  net$layers[[1]]$biases[bias.to.update] = initial.biases[bias.to.update] - eps
  net$feedForward()
  minus.loss = sum(net$reportLoss())
  
  expect_equal(
    (plus.loss - minus.loss) / 2,
    grad[[bias.to.update]]
  )
})
