context("Copying")
test_that("net$copy works", {
  x = dropoutMask(17L, 37L)
  y = dropoutMask(17L, 19L)
  
  net = mistnet(
    x,
    y,
    layer.definitions = list(
      defineLayer(
        nonlinearity = rectify.nonlinearity(), 
        size = 23, 
        prior = gaussianPrior(0, 001)
      ),
      defineLayer(
        nonlinearity = rectify.nonlinearity(), 
        size = 31, 
        prior = gaussianPrior(0, 001)
      ),
      defineLayer(
        nonlinearity = sigmoid.nonlinearity(), 
        size = ncol(y), 
        prior = gaussianPrior(0, 001)
      )
    ),
    n.ranef = 3L,
    ranefSample = gaussianRanefSample,
    n.importance.samples = 10L,
    minibatch.size = 10L,
    training.iterations = 0L,
    loss = bernoulliLoss()
  )
  
  net2 = net$copy()
  
  # Confirm that layers copy (this is the tricky one because they hide in a list
  #    object and need to be copied separately)
  
  # Modify a layer in net2
  net2$layers[[2]]$coefficients[1] = 1
  
  # net2 should change
  expect_equal(net2$layers[[2]]$coefficients[1], 1)
  
  # the original net object shouldn't
  expect_equal(net$layers[[2]]$coefficients[1], 0)
  
  
  # Confirm that normal reference classes copy correctly (i.e. shallow==FALSE)
  
  # Modify an updater in net2
  net2$layers[[2]]$bias.updater$delta[1] = 1
  
  # net2 should change
  expect_equal(net2$layers[[2]]$bias.updater$delta[1], 1)
  
  # the original net object shouldn't
  expect_equal(net$layers[[2]]$bias.updater$delta[1], 0)
  
  # Confirm that pass-by-value objects copy correctly
  
  # Modify minibatch.ids
  net2$minibatch.ids = 1:net2$minibatch.size
  
  # net2 should change
  expect_equal(net2$minibatch.ids, 1:net2$minibatch.size)
  
  # the original net object shouldn't
  expect_equal(net$minibatch.ids, numeric(0))
  
})