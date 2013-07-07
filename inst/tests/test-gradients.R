test_that("sigmoidGrad is accurate", {
  expect_equal(sigmoidGrad(s = .5), 1/4)
  expect_equal(sigmoidGrad(s = 1), 0)
  expect_equal(sigmoidGrad(s = 0), 0)
  expect_equal(
    sigmoid(1 + 1E-6) - sigmoid(1 - 1E-6), 
    sigmoidGrad(sigmoid(1)) * 2E-6
  )
})


#need to think about how the rectifier is implemented...
# test_that("rectifiedGrad is accurate", {
#   x = seq(-10, 10, by = .01)
#   expect_true(
#     all(rectifiedGrad(x, 2.01) == (x > 2.01))
#   )
#   rectify()
# })