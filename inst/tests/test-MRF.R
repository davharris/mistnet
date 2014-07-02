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
    maxit = 2L, 
    damp = damp, 
    tol = tol
  )
  
  expect_equal(mrfprob2, damped.prob)
  
  # When `all(lateral == 0)`, the output should be the same as a sigmoid
  expect_equal(
    mrf_meanfield(
      rinput=input, 
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
  importance.weights = matrix(runif(n.importance.weights * nrow), nrow = nrow)
  
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
  importance.weights = gtools::rdirichlet(nrow, rep(1, n.importance.weights))
  
  l = layer$new(
    inputs = qlogis(predicted),
    nonlinearity = mf_mrf.nonlinearity$new(
      lateral = matrix(0, nrow = ncol, ncol = ncol),
      maxit = 10L,
      damp = .2,
      tol = .01,
      l1.decay = 0,
      updater = new(
        "sgd.updater", 
        delta = matrix(0, nrow = ncol, ncol = ncol), 
        momentum = momentum,
        learning.rate = 0
      )
    )
  )
  
  y = matrix(
    rbinom(length(predicted), prob = predicted, size = 1),
    ncol = ncol(predicted)
  )
  
  l$nonlinearity$update(
    observed = y, 
    predicted = predicted,
    importance.weights = importance.weights
  )
  
  # Since learning rate was zero, nothing should happen
  expect_true(all(l$nonlinearity$lateral == 0))
  
  
  
  
  
  # Confirm that momentum works as expected.
  # No learning from the data, but a bunch of twos should get carried forward
  # from delta * momentum as 1.8s.  (The diagonal should be zeros, per usual)
  l$nonlinearity$updater$delta[,] = 2
  l$nonlinearity$update(
    observed = y, 
    predicted = predicted,
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
    l$nonlinearity$updater$delta
  )
  
  
  
  # Test a real update
  old.delta = l$nonlinearity$updater$delta
  old.lateral = l$nonlinearity$lateral
  learning.rate = 0.1
  l$nonlinearity$updater$learning.rate = learning.rate
  l$nonlinearity$update(
    observed = y, 
    predicted = predicted,
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
    old.lateral + delta
  )
  
})


test_that("MRF loss works" , {
  set.seed(1)
  
  ncol = 5 # Should be small because it goes in an exponent
  nrow = 2^ncol
  
  # Lateral should be symmetric and have zero diagonal
  lateral = matrix(rnorm(ncol^2), ncol = ncol)
  lateral = lateral + t(lateral)
  diag(lateral) = 0
  expect_equal(diag(lateral), rep(0, ncol))
  expect_true(isSymmetric(lateral))
  
  
  # All possible y vectors
  y = as.matrix(
    do.call(expand.grid, replicate(ncol, c(0, 1), simplify = FALSE))
  )
  
  y.strings = apply(y, 1, paste0, collapse = "")
  
  # Repeat the inputs across all possible y vectors
  inputs = sapply(
    1:ncol,
    function(i){
      rep(rnorm(1), nrow)
    }
  )
  
  # If lateral is zero, MRF loss should equal cross Entropy
  expect_equal(
    mrfLoss()$loss(y = y, yhat = sigmoid(inputs), lateral = 0 * lateral),
    rowSums(crossEntropy(y, sigmoid(inputs)))
  )
  
  
  
  losses = sapply(
    1:nrow,
    function(i){
      mrfLoss()$loss(y[i, ], sigmoid(inputs[i, ]), lateral)
    }
  )
  
  expect_equal(losses, mrfLoss()$loss(y, sigmoid(inputs), lateral))
  
  
  expected.frequencies = exp(-losses) / sum(exp(-losses))
  
  
  ### Estimate the frequencies by Gibbs sampling
  maxit = 500
  matches = NULL
  for(i in 1:maxit){
    for(col in 1:ncol){
      y[ , col] = rbinom(
        nrow,
        size = 1, 
        prob = sigmoid(inputs[ , col] + y %*% lateral[col, ])
      )
    }
    matches = c(matches, match(apply(y, 1, paste0, collapse = ""), y.strings))
  }
  
  # Should *not* reject the null that the samples come from the predicted 
  # distribution
  expect_true(
    .05 < chisq.test(
      x = as.integer(table(matches)), 
      p = expected.frequencies
    )$p.value
  )
})
