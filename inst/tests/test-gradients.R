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

test_that("backprop works",{
  h = matrix(rnorm(n * n.hid), nrow = n) / 10
  b2 = rnorm(n.out)
  
  w2 = w2.plus = w2.minus = matrix(rnorm(n.hid * n.out), nrow = n.hid)
  
  target.hidden = sample.int(n.hid, 1)
  
  w2.plus[target.hidden , ]  = w2[target.hidden , ] + eps / 2
  w2.minus[target.hidden , ] = w2[target.hidden , ] - eps / 2
  
  o = sigmoid(h %*% w2 %plus% b2)
  o.plus =  sigmoid(h %*% w2.plus %plus% b2)
  o.minus = sigmoid(h %*% w2.minus %plus% b2)
  
  error = crossEntropy(y = y, yhat = o)
  error.plus = crossEntropy(y = y, yhat = o.plus)
  error.minus = crossEntropy(y = y, yhat = o.minus)
  
  observed.grad = sapply(
    1:n, 
    function(i){
      (error.plus - error.minus)[i, ] / eps
    }
  )
  predicted.grad = -sapply(
    1:n, function(i){
      crossEntropyGrad(y = y, yhat = o)[i, ] * 
        sigmoidGrad(s = o)[i, ] * 
        h[i, target.hidden] 
    }
  )
  
  expect_equal(observed.grad, predicted.grad, tolerance = eps)
  
  delta = crossEntropyGrad(y = y, yhat = o) * sigmoidGrad(s = o)
  
  x = matrixMultiplyGrad(
    n.hid = n.hid, 
    n.out = n.out, 
    delta = delta,
    h = h
  )
  
  expect_equal(x[ , target.hidden], rowSums(observed.grad), tolerance = eps)
})
