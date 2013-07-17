context("Activation functions")

test_that("sigmoid produces accurate values", {
  x = matrix(seq(-10, 10, length = 1E4), ncol = 2)
  expect_equal(sigmoid(x), plogis(x))
  expect_equal(sigmoid(Inf), plogis(Inf))
  expect_equal(sigmoid(-Inf), plogis(-Inf))
})

test_that("rectify produces accurate values", {
  x = matrix(seq(-10, 10, length = 1E4), ncol = 2)
  expect_equal(rectify(x), pmax(x, 0))
})
