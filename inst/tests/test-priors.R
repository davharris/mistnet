context("Priors")

eps = 1E-5

test_that("gaussian.prior works", {
  mean = exp(1)
  var = pi
  test.points = 0:5
  
  p = gaussian.prior$new(mean = mean, var = var)
  
  # should have zero gradient at the mean
  expect_equal(p$getLogGrad(mean), 0)
  
  deltas = dnorm(test.points + eps, mean = mean, sd = sqrt(var), log = TRUE) - 
    dnorm(test.points - eps, mean = mean, sd = sqrt(var), log = TRUE)
  
  expect_equal(
    deltas/2 / eps,
    p$getLogGrad(test.points)
  )
  
  
  # should have zero gradient when variance is infinite
  p2 = gaussian.prior$new(mean = 1, var = Inf)
  expect_equal(p2$getLogGrad(1E10), 0)
})

test_that("Laplace prior works", {
  location = exp(1)
  scale = 1/pi
  rate = 1/scale
  
  p = laplace.prior$new(location = location, scale = scale)
  
  # slope shouldn't depend on distance from the median / location
  slopes = p$getLogGrad(c(location + eps, location + 2 * eps))
  expect_equal(
    slopes[[1]],
    slopes[[2]]
  )
  
  delta = log(
    dexp(location + 3 * eps, rate = rate) / 2
  ) - log(
    dexp(location + 1 * eps, rate = rate) / 2
  )
  
  expect_equal(
    delta / 2 / eps,
    p$getLogGrad(location + 2 * eps)
  )
  
})
