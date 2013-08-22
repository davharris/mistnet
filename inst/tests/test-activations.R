context("Activation functions")

test_that("sigmoid produces accurate values", {
  x = matrix(seq(-10, 10, length = 1E4), ncol = 2)
  expect_equal(sigmoid(x), plogis(x))
  expect_equal(sigmoid(Inf), plogis(Inf))
  expect_equal(sigmoid(-Inf), plogis(-Inf))
  
  y = seq(-10, 10)
  expect_equal(sigmoid(y), plogis(y))
  
  # Make sure rcpp doesn't overwrite matrices or vectors
  expect_equal(x, matrix(seq(-10, 10, length = 1E4), ncol = 2))
  expect_equal(y, seq(-10, 10))
})

test_that("rectify produces accurate values", {
  x = matrix(seq(-10, 10, length = 1E4), ncol = 2)
  
  expect_equal(rectify(x), pmax(x, 0))
  
  # Sometimes (but maybe not always?) an old version of the rcpp rectifier would
  # overwrite x.  This makes sure x hasn't been rectified
  expect_equal(x, matrix(seq(-10, 10, length = 1E4), ncol = 2))
})
