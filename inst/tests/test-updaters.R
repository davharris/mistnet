context("Updaters")

test_that("sgd updater works",{
  
  set.seed(1)
  coefs = matrix(0, 10, 10)
  true.coefs = structure(rnorm(length(coefs)), dim = dim(coefs))
  
  updater = new(
    "sgd.updater", 
    delta = coefs, 
    momentum = .9,
    learning.rate = .01
  )
  
  for(i in 1:1000){
    updater$computeDelta(coefs - true.coefs)
    coefs = coefs + updater$delta
  }
  
  expect_equal(coefs, true.coefs) 
})


test_that("adagrad works",{
  
  set.seed(1)
  coefs = matrix(0, 10, 10)
  true.coefs = structure(rnorm(length(coefs)), dim = dim(coefs))
  
  updater = new(
    "adagrad.updater", 
    delta = coefs, 
    squared.grad = coefs,
    learning.rate = 1
  )
  
  for(i in 1:1000){
    updater$computeDelta(coefs - true.coefs)
    coefs = coefs + updater$delta
  }
  
  expect_equal(coefs, true.coefs)
  
})

test_that("adadelta works",{
  
  set.seed(1)
  coefs = matrix(0, 10, 10)
  true.coefs = structure(rnorm(length(coefs)), dim = dim(coefs))
  
  updater = new(
    "adadelta.updater", 
    delta = coefs, 
    rho = .95, 
    epsilon = 1E-6
  )
  
  for(i in 1:1000){
    updater$computeDelta(coefs - true.coefs)
    coefs = coefs + updater$delta
  }
  
  expect_equal(coefs, true.coefs)
  
})
