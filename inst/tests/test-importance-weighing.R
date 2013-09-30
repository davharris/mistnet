context("Importance weighing")

test_that("Importance weighting works", {
  nrow = 133
  ncol = 237
  
  # rows are data points; columns are samples.
  # unscaled probabilities from a half-Cauchy should ensure some extreme weights
  outcomes = abs(matrix(rt(nrow * ncol, df = 1), nrow = nrow, ncol = ncol)) * 10
  true.probs = outcomes / rowSums(outcomes)
  
  # each row can have a different baseline log-probability
  # Set a very high mean error (mean 100) to make sure it works in this area
  raw.errors = -log(true.probs) + rnorm(nrow, mean = 100)
  
  expect_equal(weighImportance(raw.errors), true.probs)
})
