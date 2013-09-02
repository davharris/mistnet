context("Minibatch generation")
test_that("minibatch generation works",{
  n = 1001L
  niter = n
  set.seed(1)
  
  net = network$new(
    x = matrix(1:n, nrow = n),
    minibatch.size = n + 1L
  )
  expect_error(net$selectMinibatch(), "replace = FALSE")
  
  
  net$minibatch.size = -1L
  expect_error(net$selectMinibatch(), "invalid 'size' argument")
  
  
  net$minibatch.size = n
  net$selectMinibatch()
  expect_true(all(table(net$minibatch.ids) == 1))
  
  
  net$minibatch.size = as.integer(n/2)
  net$selectMinibatch()
  expect_equal(length(net$minibatch.ids), net$minibatch.size)
  
  
  counts = integer(n)
  for(i in 1:niter){
    net$selectMinibatch()
    counts[net$minibatch.ids] = counts[net$minibatch.ids] + 1
  }
  expect_equal(
    mean(counts),
    net$minibatch.size/n * niter
  )
  expect_true(
    var(counts) < 300
  )
})