# Still need to test:
# Tolerance behaves as expected

context("MRF")
test_that("mean field MRF works", {
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
})

