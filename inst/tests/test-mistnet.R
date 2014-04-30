context("Mistnet function")

x = dropoutMask(17L, 37L)
y = dropoutMask(17L, 19L)

colnames(y) = letters[1:ncol(y)]

test_that("Correct behavior for fewer than one iteration",{
  net = mistnet(
    x,
    y,
    nonlinearity.names = c("sigmoid", "rectify", "sigmoid"),
    hidden.dims = c(5L, 7L),
    n.ranef = 3L,
    ranefSample = gaussianRanefSample,
    n.importance.samples = 10L,
    minibatch.size = 10L,
    training.iterations = 0L,
    loss.name = "crossEntropy"
  )
  
  expect_equal(net$completed.iterations, 0)
  expect_error(net$fit(-1), "valid number of iterations")
  
  net$fit(2) # shouldn't throw an error
  
  expect_equal(dimnames(net$layers[[3]]$outputs)[[2]], colnames(y))
  expect_equal(colnames(net$layers[[3]]$coefficients), colnames(y))
})
