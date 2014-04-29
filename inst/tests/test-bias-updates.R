context("Bias updates")

test_that("biases update correctly",{
  
  eps = 1E-5
  x = matrix(rnorm(1000), nrow = 50)
  y = dropoutMask(50, 17)
  minibatch.size = 11L
  n.importance.samples = 2L
  
  net = mistnet(
    x = x,
    y = y,
    nonlinearity.names = c("sigmoid"),
    priors = list(gaussian.prior$new(mean = 0, var = 1)),
    hidden.dims = NULL,
    n.ranef = 3L,
    minibatch.size = minibatch.size,
    n.importance.samples = 5L,
    loss = crossEntropy,
    lossGrad = crossEntropyGrad,
    ranefSample = gaussianRanefSample,
    training.iterations = 0L
  )
  
  initial.biases = rnorm(ncol(y))
  net$layers[[1]]$biases = matrix(initial.biases, nrow = 1)
    
  net$fit(1)
  
  # Should actually do what the updater says
  expect_equal(
    net$layers[[1]]$biases, 
    initial.biases + net$layers[[1]]$bias.updater$delta
  )
  
  # Delta should be calculated correctly.
  # Divide by minibatch size to standardize the values regardless of # of examples
  with(
    net$layers[[1]],
    expect_equal(
      weighted.bias.grads / minibatch.size * bias.updater$learning.rate,
      -c(bias.updater$delta)
    )
  )
  
  #################
  # Tests below this line haven't been updated
  
  # Reset coefficients to zero
  net$layers[[1]]$coefficients[ , ] = 0
  
  # With reset coefficients, let's see what the loss is.
  net$feedForward(1L)
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
