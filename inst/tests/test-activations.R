test_that("sigmoid produces accurate values", {
  x = matrix(seq(-10, 10, length = 1E4), ncol = 2)
  expect_equal(sigmoid(x), plogis(x))
})

test_that("rectify produces accurate values", {
  x = matrix(seq(-10, 10, length = 1E4), ncol = 2)
  expect_equal(rectify(x), pmax(x, 0))
})