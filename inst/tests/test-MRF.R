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

