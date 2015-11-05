context("Bias updates")

test_that("biases update correctly",{
  
  eps = 1E-5
  x = matrix(rnorm(1000), nrow = 50)
  y = dropoutMask(50, 17)
  n.minibatch = 11L
  n.importance.samples = 2L
  
  net = mistnet(
    x = x,
    y = y,
    layer.definitions = list(
      defineLayer(
        nonlinearity = sigmoid.nonlinearity(), 
        size = ncol(y), 
        prior = gaussian.prior(mean = 0, sd = sqrt(.1))
      )
    ),
    sampler = gaussian.sampler(ncol = 1L, sd = 1),
    loss = bernoulliLoss(),
    updater = sgd.updater(learning.rate = .01, momentum = .9),
    n.minibatch = n.minibatch
  )
  
  initial.biases = rnorm(ncol(y))
  net$layers[[1]]$biases = matrix(initial.biases, nrow = 1)
    
  net$fit(1)
  
  # Confirm weighted bias gradients
  
  net$layers[[1]]$weighted.bias.grads
  
  
  # Should actually do what the updater says
  expect_equal(
    net$layers[[1]]$biases, 
    initial.biases + net$layers[[1]]$bias.updater$delta
  )
  
  # Delta should be calculated correctly.
  # Divide by n.minibatch to standardize the values regardless of # of examples
  with(
    net$layers[[1]],
    expect_equal(
      weighted.bias.grads / n.minibatch * bias.updater$learning.rate,
      -c(bias.updater$delta)
    )
  )
  
})
