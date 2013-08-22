context("Layer creation")

test_that("Assertions are respected in createLayer", {
  f = function(learning.rate, momentum){
    createLayer(
      dim = c(1L, 2L),
      learning.rate = learning.rate,
      momentum = momentum,
      prior = gaussian.prior$new(),
      dataset.size = 7,
      nonlinearity.name = "sigmoid"
    )
  }
  
  # expect no error with reasonable values
  f(learning.rate = 0.01, momentum = .9)
  f(learning.rate = 0.01, momentum = 0)
  
  # Bad learning rates:
  expect_error(
    f(learning.rate = 0.0, momentum = .9),
    "learning.rate"
  )
  expect_error(
    f(learning.rate = 2, momentum = .9),
    "learning.rate"
  )
  expect_error(
    f(learning.rate = -.9, momentum = .9),
    "learning.rate"
  )
  
  
  # Bad momentum
  expect_error(
    f(learning.rate = .01, momentum = 1),
    "momentum"
  )
  expect_error(
    f(learning.rate = .01, momentum = 2),
    "momentum"
  )
  expect_error(
    f(learning.rate = .01, momentum = -1),
    "momentum"
  )
})
