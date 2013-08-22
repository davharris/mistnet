context("Dropout")

test_that("Dropout masks work", {
  set.seed(1)
  mask = dropoutMask(1E4, 1E3)
  
  expect_equal(dim(mask), c(1E4, 1E3))
  
  expect_equal(.5, mean(mask), tolerance = 1E-4)
  expect_true(var(colMeans(mask)) < 1E-4)
  expect_true(var(colMeans(mask)) < 1E-4)
})

