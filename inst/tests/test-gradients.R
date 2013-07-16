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
n.hid = 13
n.out = 17
h = matrix(rnorm(n.hid * n), nrow = n)

w2 = matrix(rnorm(n.hid * n.out), nrow = n.hid, ncol = n.out)
w2.plus = w2 + eps
w2.minus = w2 - eps

o = sigmoid(h %*% w2)
o.plus = sigmoid(h %*% w2.plus)
o.minus = sigmoid(h %*% w2.minus)

delta = 0.5 / eps * (o.plus - o.minus)

grad = sigmoidGrad(s = o) * repvec(rowSums(h), ncol(w2))

all.equal(delta, grad)