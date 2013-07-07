context("Adding biases")

test_that("biases add correctly", {
  x = matrix(rnorm(150), ncol = 10)
  b = rnorm(ncol(x))
  expect_equal(
    x %plus% b,
    t(t(x) + b)
  )
})