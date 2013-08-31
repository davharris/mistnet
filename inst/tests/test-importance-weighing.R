context("Importance weighing")

test_that("Importance weighting works", {
  nrow = 17
  ncol = 29
  
  # rows are data points; columns are samples.
  # unscaled probabilities from a half-Cauchy should ensure some extreme weights
  outcomes = abs(matrix(rt(nrow * ncol, df = 1), nrow = nrow, ncol = ncol))
  true.probs = outcomes / rowSums(outcomes)
  
  # each row can have a different baseline log-probability
  raw.errors = -log(true.probs) + rnorm(nrow)
  
  expect_equal(weighImportance(raw.errors), true.probs)
})
