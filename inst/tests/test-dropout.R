context("Dropout")

test_that("Dropout masks work", {
  set.seed(1)
  mask = dropoutMask(1E4, 1E3)
  
  expect_equal(dim(mask), c(1E4, 1E3))
  
  expect_equal(.5, mean(mask), tolerance = 1E-4)
  expect_true(var(colMeans(mask)) < 1E-4)
  expect_true(var(colMeans(mask)) < 1E-4)
})

test_that("feedforward works with dropout", {
  l = layer$new(
    biases = 1:7,
    coefficients = matrix(rnorm(28), nrow = 4),
    nonlinearity = rectify,
    dim = c(4L, 7L),
    dropout = TRUE
  )
  
  input.matrix = matrix(rnorm(20), ncol = 4)
  l$forwardPass(input.matrix)
  
  
  expect_equal(
    l$input,
    input.matrix
  )
  expect_equal(
    l$activation,
    (l$input %*% l$coefficients) %plus% l$biases
  )
  
  is.unchanged = l$nonlinearity(l$activation) == l$output
  is.zero = l$output == 0
  expect_true(all(is.zero | is.unchanged))
})
