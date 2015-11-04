context("Mistnet function")

x = dropoutMask(17L, 37L)
y = dropoutMask(17L, 19L)

colnames(y) = letters[1:ncol(y)]

test_that("Correct behavior for fewer than one iteration",{
  net = mistnet(
    x,
    y,
    layer.definitions = list(
      defineLayer(rectify.nonlinearity(), 10, gaussian.prior(mean = 0, sd = 1)),
      defineLayer(rectify.nonlinearity(), 11, gaussian.prior(mean = 0, sd = 1)),
      defineLayer(sigmoid.nonlinearity(), ncol(y), gaussian.prior(mean = 0, sd = 1))
    ),
    n.importance.samples = 10L,
    n.minibatch = 10L,
    training.iterations = 0L,
    loss = bernoulliLoss(),
    updater = adagrad.updater(learning.rate = .01)
  )
  
  expect_equal(net$completed.iterations, 0)
  expect_error(net$fit(-1), "valid number of iterations")
  
  net$fit(2) # shouldn't throw an error
  
  expect_equal(dimnames(net$layers[[3]]$outputs)[[2]], colnames(y))
  expect_equal(colnames(net$layers[[3]]$weights), colnames(y))
})
