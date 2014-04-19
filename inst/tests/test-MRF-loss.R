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
  
  losses = sapply(
    1:nrow,
    function(i){
      mrfLoss(y[i, ], sigmoid(inputs[i, ]), lateral)
    }
  )
  
  expected.frequencies = exp(-losses) / sum(exp(-losses))
  
  
  ### Estimate the frequencies by Gibbs sampling
  maxit = 2500
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
