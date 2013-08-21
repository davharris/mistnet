context("Feedforward")

test_that("Single-layer feedforward works", {
  l = layer$new(
    biases = 1:7,
    coefficients = matrix(rnorm(28), nrow = 4),
    nonlinearity = rectify
  )
  l.copy = l$copy()
  
  input.matrix = matrix(rnorm(20), ncol = 4)
  l$forwardPass(input.matrix)
  
  
  expect_equal(
    l$input,
    input.matrix
  )
  expect_equal(
    l$output,
    l$nonlinearity((l$input %*% l$coefficients) %plus% l$biases)
  )
  
  # copy shouldn't be overwritten by the forward pass on l
  expect_equal(length(l.copy$input), 0)
  
  # copy's only difference from original should be the input and output fields
  l.copy$input = l$input
  l.copy$output = l$output
  expect_equal(l, l.copy)
})
