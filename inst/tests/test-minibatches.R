context("Minibatches")
test_that("minibatches work", {
  set.seed(1)
  
  x = matrix(rnorm(5 * 7), nrow = 5)
  y = dropoutMask(n.row = 5, n.col = 11)
  
  net = mistnet(
    x = x,
    y = y,
    layer.definitions = list(
      defineLayer(
        nonlinearity = sigmoid.nonlinearity(), 
        size = ncol(y), 
        prior = gaussian.prior(mean = 0, sd = 1)
      )
    ),
    loss = bernoulliLoss(),
    n.minibatch = 4L,
    shuffle = TRUE
  )
  
  tally = rep(0L, nrow(x))
  
  
  
  for(i in 1:20){
    net$fit(1)
    # All elements should be selected equally often, +/- 1
    expect_true(max(tally) <= (min(tally) + 1))
    for(j in net$row.selector$minibatch.ids){
      tally[j] = tally[j] + 1
    }
  }
  
  # Should go through the correct number of epochs
  expect_equal(
    (i * net$row.selector$n.minibatch) %/% nrow(x),
    min(tally)
  )
  
  # Number of epochs should be counted correctly
  expect_equal(
    net$row.selector$completed.epochs,
    min(tally)
  )
  
  expect_false(
    all(
      net$row.selector$minibatch.ids == seq_len(net$row.selector$n.minibatch)
    )
  )
  
  
  expect_error(
    mistnet(
      x = x,
      y = y,
      layer.definitions = list(
        defineLayer(
          nonlinearity = sigmoid.nonlinearity(), 
          size = ncol(y), 
          prior = gaussian.prior(mean = 0, var = 1)
        )
      ),
      loss = bernoulliLoss(),
      n.minibatch = 0L,
      shuffle = TRUE
    ),
    regexp = "^.*not greater than 0.*$"
  )
})
