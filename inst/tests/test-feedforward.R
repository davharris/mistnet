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
    l$activation,
    (l$input %*% l$coefficients) %plus% l$biases
  )
  expect_equal(
    l$output,
    l$nonlinearity((l$input %*% l$coefficients) %plus% l$biases)
  )
  
  # copy shouldn't be overwritten by the forward pass on l
  # (making sure that I didn't accidentally cheat when I made the copy)
  expect_equal(length(l.copy$input), 0)
  
  # copy's only difference from original should be the input and output fields
  l.copy$input = l$input
  l.copy$output = l$output
  expect_equal(l, l.copy)
})


test_that("Multi-layer feedforward works", {
  net = network$new(
    x = matrix(rnorm(100), nrow = 20, ncol = 5),
    layers = list(
      l1 = layer$new(
        biases = rnorm(6),
        coefficients = matrix(rnorm(30), nrow = 5),
        nonlinearity = rectify,
        dim = c(5L, 6L)
      ),
      l2 = layer$new(
        biases = rnorm(7),
        coefficients = matrix(rnorm(42), nrow = 6),
        nonlinearity = sigmoid
      )
    ),
    n.layers = 2L,
    minibatch.size = 5L
  )
  
  
  net$newMinibatch()
  net$feedForward()
  
  expect_equal(
    with(
      net$layers[[1]],
      nonlinearity((input %*% coefficients) %plus% biases)
    ),
    net$layers[[1]]$output
  )
  
  expect_equal(
    net$layers[[1]]$output,
    net$layers[[2]]$input
  )
  
  expect_equal(
    with(
      net$layers[[2]],
      nonlinearity((input %*% coefficients) %plus% biases)
    ),
    net$layers[[2]]$output
  )
})
