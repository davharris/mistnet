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

test_that("Poisson loss is correct",{
  N = 1E3
  yhat = matrix(rexp(N), ncol = 10)
  y = matrix(rpois(N, 5), ncol = 10)
  expect_equal(
    poissonLoss(y = y, yhat = yhat),
    - dpois(x = y, lambda = yhat, log = TRUE)
  )
  
  expect_equal(dim(poissonLoss(y = y, yhat = yhat)), dim(yhat))
  
  # No other tests should be needed; dpois is already tested by R-core.
})

test_that("Squared loss is correct",{
  N = 1E3
  yhat = matrix(rnorm(N), ncol = 10)
  y = matrix(rnorm(N), ncol = 10)
  expect_equal(
    squaredLoss(y = y, yhat = yhat),
    (y - yhat)^2
  )
  
  expect_equal(dim(squaredLoss(y = y, yhat = yhat)), dim(yhat))
})

