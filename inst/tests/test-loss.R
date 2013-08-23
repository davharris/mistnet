context("Loss functions")

test_that("Cross-entropy is correct",{
  y = dropoutMask(5, 6)
  yhat = sigmoid(rnorm(length(y)))
  expect_equal(
    crossEntropy(y = y, yhat = yhat),
    -dbinom(x = y, prob = yhat, size = 1, log = TRUE)
  )
  
  expect_equal(
    crossEntropy(y = 0, yhat = 1),
    Inf
  )
  
  crossEntropyOfCertainty = crossEntropy(y = y, yhat = 0)
  expect_true(
    all(is.na(crossEntropyOfCertainty) | crossEntropyOfCertainty == Inf)
  )
})