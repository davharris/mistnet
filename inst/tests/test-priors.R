context("Priors")

eps = 1E-5

test_that("gaussian.prior works", {
  mean = exp(1)
  sd = sqrt(pi)
  test.points = 0:5
  
  p = gaussian.prior(mean = mean, sd = sd)
  
  # should have zero gradient at the mean
  expect_equal(p$getLogGrad(mean), 0)
  
  deltas = dnorm(test.points + eps, mean = mean, sd = sd, log = TRUE) - 
    dnorm(test.points - eps, mean = mean, sd = sd, log = TRUE)
  
  expect_equal(
    deltas/2 / eps,
    p$getLogGrad(test.points)
  )
  
  
  # should have zero gradient when variance is infinite
  p2 = gaussian.prior(mean = 1, sd = Inf)
  expect_equal(p2$getLogGrad(1E10), 0)
})

test_that("Laplace prior works", {
  location = exp(1)
  scale = 1/pi
  rate = 1/scale
  
  p = laplace.prior(location = location, scale = scale)
  
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


test_that("GP prior works", {
  dim = 2
  
  x = seq(1, 7, length = 50)
  
  truth = sin(x)
  
  y = matrix(rnorm(dim * length(x), sd = .1), nrow = dim)
  y = t(t(y) + truth)
  y[1, ] = -y[1, ]
  y = t(scale(t(y)))

  lengthscale = c(pi, exp(1))
  
  
  # kernlab solution
  var = c(.02, .04)
  sigma = 1/2/lengthscale^2
  
  gp1 = kernlab::gausspr(x = x, y = y[1, ], variance.model = TRUE, scaled = FALSE, var = var[1], kpar = list(sigma = sigma[1]), tol = 1E-6)
  gp2 = kernlab::gausspr(x = x, y = y[2, ], variance.model = TRUE, scaled = FALSE, var = var[2], kpar = list(sigma = sigma[2]), tol = 1E-6)
  
  
  # mistnet solution
  K = array(0, dim = c(length(x), length(x), dim))
  d_squared = as.matrix(dist(x))^2
  
  K[,,1] = exp(-d_squared * sigma[1]) # sigma is 0.5/lengthscale^2
  K[,,2] = exp(-d_squared * sigma[2])
  
  prior = gp.prior$new(K = K, coefs = y, noise_sd = sqrt(var))

  expect_equal(c(gp1@fitted), c(prior$means[1, ]))
  expect_equal(c(gp2@fitted), c(prior$means[2, ]))
  
  # Posterior variance should equal function variance plus residual variance
  expect_equal(
    diag(prior$posterior_var[ , , 1]),
    c(predict(gp1, x, type = "variance")) + var[1]
  )

  expect_equal(
    diag(prior$posterior_var[ , , 2]),
    c(predict(gp2, x, type = "variance")) + var[2]
  )
  
  # several variables should be stored as-is
  expect_equal(prior$noise_sd, sqrt(var))
  expect_equal(K, prior$K)
  
  for(i in 1:dim){
    # inverse_var is the matrix inverse of posterior_var
    expect_equal(prior$inverse_var[,,i], solve(prior$posterior_var[,,i]))
  }  
  
  
  # Should also test that all this works with a one-dimensional prior...
  
  
  # Test the gradient numerically by setting adding epsilon to the first element of y
  # and seeing how much the log-likelihood changes
  eps = 0 * y[1, ]
  eps[1] = 1E-6
  ll_plus = mvtnorm::dmvnorm(y[1, ] + eps, prior$means[1, ], prior$posterior_var[ , , 1], log = TRUE)
  ll_minus = mvtnorm::dmvnorm(y[1, ] - eps, prior$means[1, ], prior$posterior_var[ , , 1], log = TRUE)

  grad_est = (ll_plus - ll_minus) / 2 / eps[1]
  
  grad = prior$inverse_var[,,1] %*% (y[1, ] - prior$means[1, ])

  # (minus because grad is for negative log likelihood)
  expect_equal(-grad[1], grad_est)
})
