context("Backpropagation")

test_that("Backpropagation works",{

eps = 1E-5

net = network$new(
  x = matrix(rnorm(99), nrow = 33, ncol = 3),
  y = dropoutMask(20, 7),
  layers = list(
    l1 = createLayer(
      dim = c(3L, 6L),
      learning.rate = .01,
      momentum = .8,
      prior = gaussian.prior$new(mean = 0, var = 1),
      dataset.size = 1000,
      nonlinearity.name = "sigmoid",
      dropout = FALSE
    ),
    l2 = createLayer(
      dim = c(6L, 7L),
      learning.rate = .01,
      momentum = .8,
      prior = gaussian.prior$new(mean = 0, var = 1),
      dataset.size = 1000,
      nonlinearity.name = "sigmoid",
      dropout = FALSE
    )
  ),
  n.layers = 2L,
  minibatch.size = 15L,
  loss = crossEntropy,
  lossGradient = crossEntropyGrad
)


net$newMinibatch()
net$feedForward()
net$backprop()

# If all the coefficients in layer 2 are 0, then the
# weights in layer 1 can't possibly matter
expect_true(all(net$layers[[1]]$llik.grad == 0))


# Set coefficients to random values
net$layers[[1]]$coefficients[ , ] = rnorm(net$layers[[1]]$coefficients)
net$layers[[2]]$coefficients[ , ] = rnorm(net$layers[[2]]$coefficients)

# Compare the computed gradient for layer 2 with a finite difference 
# approximation
net$feedForward()
net$backprop()
grad = net$layers[[2]]$llik.grad

net$layers[[2]]$coefficients[1,1] = eps + net$layers[[2]]$coefficients[1,1]
net$feedForward()
net$backprop()
plus.error = net$loss(
  y = net$y[net$minibatch.ids, ], 
  yhat = net$layers[[2]]$output
)

# subtract 2 * eps: once to undo the addition above and once to actually
# decrement by eps
net$layers[[2]]$coefficients[1,1] = -2 * eps + net$layers[[2]]$coefficients[1,1]
net$feedForward()
net$backprop()
minus.error = net$loss(
  y = net$y[net$minibatch.ids, ], 
  yhat = net$layers[[2]]$output
)

expect_equal(
  sum((plus.error - minus.error)) / 2 / eps,
  grad[1,1]
)
})
