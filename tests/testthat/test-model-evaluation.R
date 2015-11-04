context("Model evaluation")

test_that("logMeanExp works", {
  set.seed(1)
  loglik = rnorm(1E6, sd = 100)
  w = sample(abs(loglik) / sum(abs(loglik)))
  
  # Bizarrely, there doesn't seem to be any floating point error
  # Well, I'm not going to look a gift horse in the mouth...
  expect_equal(
    logMeanExp(loglik, w),
    log(sum(exp(loglik) * w))
  )
  
  # Make sure it works when we'd get overflow
  loglik = rep(1000, 1000)
  
  expect_equal(log(mean(exp(loglik))), Inf)
  expect_equal(logMeanExp(loglik), 1000)
})


