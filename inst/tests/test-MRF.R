# Still need to test:
# Tolerance behaves as expected

context("MRF")
test_that("mean field MRF feeds forward", {
  input = matrix(rnorm(110), nrow = 11, ncol = 10)
  lateral = matrix(rnorm(100), nrow = 10, ncol = 10)
  diag(lateral) = 0
  maxit = 1L
  damp = 0.8
  tol = 0
  
  
  # The function should match this process for the first iteration
  startprob = sigmoid(input)
  prob = sigmoid(startprob %*% lateral + input)
  damped.prob = startprob * damp + prob * (1 - damp)
  
  mrfprob = mrf_meanfield(
    rinput=input, 
    rlateral=lateral, 
    rprob = sigmoid(input),
    maxit = maxit, 
    damp = damp, 
    tol = tol
  )
  
  expect_equal(mrfprob, damped.prob)
  
  
  
  
  # Try a second iteration
  prob = sigmoid(damped.prob %*% lateral + input)
  damped.prob = damped.prob * damp + prob * (1 - damp)
  
  mrfprob2 = mrf_meanfield(
    rinput=input, 
    rlateral=lateral, 
    rprob = sigmoid(input),
    maxit = 2L, 
    damp = damp, 
    tol = tol
  )
  
  expect_equal(mrfprob2, damped.prob)
  
  # When `all(lateral == 0)`, the output should be the same as a sigmoid
  expect_equal(
    mrf_meanfield(
      rinput=input, 
      rprob = sigmoid(input),
      rlateral=lateral * 0, 
      maxit = 200L, 
      damp = damp, 
      tol = tol
    ),
    sigmoid(input)
  )
})



test_that("findPredictedCrossprod works", {
  nrow = 7
  ncol = 11
  n.importance.weights = 1
  
  # If there's just one importance sample, it should return the same as
  # crossprod.
  predicted = array(
    runif(nrow * ncol * n.importance.weights),
    dim = c(nrow, ncol, n.importance.weights)
  )
  importance.weights = matrix(rep(1, nrow), ncol = 1)
  
  expect_equal(
    findPredictedCrossprod(predicted, importance.weights),
    crossprod(predicted[,,1])
  )
  
  # Test with three samples ##############################
  n.importance.weights = 3
  predicted = array(
    runif(nrow * ncol * n.importance.weights),
    dim = c(nrow, ncol, n.importance.weights)
  )
  importance.weights = rdirichlet(nrow, rep(1, n.importance.weights))
  
  x = findPredictedCrossprod(predicted, importance.weights)
  
  x2 = x * 0
  for(i in 1:nrow){
    for(j in 1:n.importance.weights){
      x2 = x2 + tcrossprod(predicted[i, , j]) * importance.weights[i, j]
    }
  }
  
  expect_equal(x, x2)
})

test_that("lateral updates work", {
  nrow = 7
  ncol = 11
  n.importance.weights = 1
  momentum = 0.75
  
  # If there's just one importance sample, it should return the same as
  # crossprod.
  predicted = array(
    runif(nrow * ncol * n.importance.weights),
    dim = c(nrow, ncol, n.importance.weights)
  )
  importance.weights = rdirichlet(nrow, rep(1, n.importance.weights))
  
  l = layer$new(
    activations = qlogis(predicted),
    nonlinearity = mf_mrf.nonlinearity$new(
      lateral = matrix(0, nrow = ncol, ncol = ncol),
      maxit = 10L,
      damp = .2,
      tol = .01,
      delta = matrix(0, nrow = ncol, ncol = ncol),
      l1.decay = 0
    )
  )
  
  y = matrix(
    rbinom(length(predicted), prob = predicted, size = 1),
    ncol = ncol(predicted)
  )
  
  l$nonlinearity$update(
    observed = y, 
    predicted = predicted,
    learning.rate = 0,
    momentum = 0,
    importance.weights = importance.weights
  )
  
  # Since learning rate was zero, nothing should happen
  expect_true(all(l$nonlinearity$lateral == 0))
  
  
  
  
  
  # Confirm that momentum works as expected.
  # No learning from the data, but a bunch of twos should get carried forward
  # from delta * momentum as 1.8s.  (The diagonal should be zeros, per usual)
  l$nonlinearity$delta[,] = 2
  l$nonlinearity$update(
    observed = y, 
    predicted = predicted,
    learning.rate = 0,
    momentum = momentum,
    importance.weights = importance.weights
  )
  
  expect_equal(
    l$nonlinearity$lateral,
    2 * momentum * (1 - diag(ncol))
  )
  
  # With no learning and initialization with no lateral coefficients, lateral
  # should come straight from delta.
  expect_equal(
    l$nonlinearity$lateral,
    l$nonlinearity$delta
  )
  
  
  
  # Test a real update
  old.delta = l$nonlinearity$delta
  old.lateral = l$nonlinearity$lateral
  learning.rate = 0.01
  l$nonlinearity$update(
    observed = y, 
    predicted = predicted,
    learning.rate = learning.rate,
    momentum = momentum,
    importance.weights = importance.weights
  )
  
  
  diff = crossprod(y) - findPredictedCrossprod(
    predicted = predicted, 
    importance.weights = importance.weights
  )
  
  delta = old.delta * momentum + diff * learning.rate / nrow
  diag(delta) = 0
  expect_equal(
    l$nonlinearity$lateral,
    delta + old.lateral
  )
  
})
