context("Evaluation")
test_that("LogMeanExp works",{
  
  expect_equal(logMeanExp(1:5), log(mean(exp(1:5))))
  
  big = 10^(seq(0, 20, length = 10))
  expect_equal(logMeanExp(big), BayesFactor::logMeanExpLogs(big))
  
  small = 10^(seq(0, -20, length = 10))
  expect_equal(logMeanExp(small), BayesFactor::logMeanExpLogs(small))
})

