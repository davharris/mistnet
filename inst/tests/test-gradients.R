context("Gradients")

test_that("sigmoidGrad is accurate", {
  expect_equal(sigmoidGrad(s = .5), 1/4)
  expect_equal(sigmoidGrad(s = 1), 0)
  expect_equal(sigmoidGrad(s = 0), 0)
  expect_equal(
    sigmoid(1 + 1E-6) - sigmoid(1 - 1E-6), 
    sigmoidGrad(sigmoid(1)) * 2E-6
  )
})


test_that("rectifiedGrad is accurate", {
  x = matrix(seq(-10, 10, length = 1E3), ncol = 2)
  expect_true(
    all(rectifiedGrad(x) == (x > 0))
  )
  expect_equal(
    rectifiedGrad(x) * 2E-6,
    rectify(x + 1E-6) - rectify(x - 1E-6)
  )
})



eps = 1E-7
n = 37
n.out = 17
n.hid = 13
y = matrix(rbinom(n * n.out, size = 1, prob = .5), ncol = n.out)

test_that("output layer gradient is accurate", {
  
  # activation
  a = matrix(rnorm(n * n.out), nrow = n)
  
  # output
  o = sigmoid(a)
  o.plus = sigmoid(a + eps)
  o.minus = sigmoid(a - eps)
  
  error = crossEntropy(y = y, yhat = o)
  error.plus = crossEntropy(y = y, yhat = o.plus)
  error.minus = crossEntropy(y = y, yhat = o.minus)
  
  error.grad = .5 / eps * (error.plus - error.minus)
  
  expect_equal(
    - crossEntropyGrad(y = y, yhat = o) * sigmoidGrad(s = o),
    error.grad
  )
})

test_that("gradients can be passed through matrix multiplies",{
  h = matrix(rnorm(n * n.hid), nrow = n)
  b2 = rnorm(n.out)
  
  w2 = matrix(rnorm(n.hid * n.out), nrow = n.hid)
  w2.plus  = w2 + eps
  w2.minus = w2 - eps
  
  a = h %*% w2 %plus% b2
  a.plus =  h %*% w2.plus %plus% b2
  a.minus = h %*% w2.minus %plus% b2
  
  grad = .5/eps * (a.plus - a.minus)
  
  x = repvec(rowSums(h), n.out)
  
  expect_equal(grad, x)
})

