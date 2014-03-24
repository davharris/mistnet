test_that("minibatches work", {
  
  net = network$new(
    completed.iterations = 0L,
    minibatch.size = 29L,
    x = matrix(NA, nrow = 51, ncol = 33)
  )
  net$selectMinibatch()
  
  # First minibatch should bet the first bunch of numbers
  expect_equal(net$minibatch.ids, 1:net$minibatch.size)
  
  # Next minibatch should wrap around
  net$completed.iterations = 1L
  net$selectMinibatch()
  
  # Should hit the largest value before looping around
  expect_equal(max(net$minibatch.ids), nrow(net$x))
  
  # Should start over at 1
  expect_equal(min(net$minibatch.ids), 1L)
  
  # Should have the correct size even after looping
  expect_equal(length(net$minibatch.ids), net$minibatch.size)
  
  
  
  # Minibatch size must be positive
  net$minibatch.size = -1L
  expect_error(net$selectMinibatch())
  
  net$minibatch.size = 0L
  expect_error(net$selectMinibatch())
  
  
  # If minibatch.size = nrow(x), all rows should be sampled once.
  net$minibatch.size = nrow(net$x)
  net$selectMinibatch()
  expect_true(all(table(net$minibatch.ids) == 1))
  
  
  
  # After looping around N times, everone row should occur once per minibatch.size
  net$minibatch.size = 19L
  counts = integer(nrow(net$x))
  for(i in 1:nrow(net$x)){
    net$completed.iterations = net$completed.iterations + 1L
    net$selectMinibatch()
    counts[net$minibatch.ids] = counts[net$minibatch.ids] + 1
  }
  expect_equal(
    counts,
    rep(net$minibatch.size, nrow(net$x))
  )
})
