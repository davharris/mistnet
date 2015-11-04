context("Adding biases")

test_that("biases add correctly", {
  x = matrix(seq(-10, 10, length = 70), ncol = 10)
  b = 1:ncol(x)
  
  # biases should add like the recycling rule, but transposed
  expect_equal(
    x %plus% b,
    t(t(x) + b)
  )
  
  expect_equal(x, matrix(seq(-10, 10, length = 70), ncol = 10))
  expect_equal(b, 1:ncol(x))
})
