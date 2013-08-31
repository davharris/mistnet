context("Feedforward")

test_that("Single-layer feedforward works", {
  l = createLayer(
    n.inputs = 4L,
    n.outputs = 7L,
    prior = gaussian.prior$new(mean = 0, var = 1),
    nonlinearity.name = "sigmoid",
    minibatch.size = 5L,
    n.importance.samples = 3L
  )
  l.copy = l$copy()
  
  input.matrix = matrix(rnorm(20), ncol = 4)
  expect_error(l$forwardPass(input.matrix), "sample.num is missing")
  
  l$forwardPass(input.matrix, 2L)
  
  expect_equal(
    l$inputs[ , , 2],
    input.matrix
  )
  expect_true(
    all(is.na(l$inputs[ , , -2]))
  )
  
  
  expect_equal(
    l$activations[ , , 2],
    (l$inputs[ , , 2] %*% l$coefficients) %plus% l$biases
  )
  expect_equal(
    l$outputs[ , , 2],
    l$nonlinearity((l$inputs[ , , 2] %*% l$coefficients) %plus% l$biases)
  )

  # Nothing should change during feedforward except the three listed fields
  for(name in layer$fields()){
    name.shouldnt.change = name %in% c("inputs", "activations", "outputs")
    if(name.shouldnt.change){
    }else{
      expect_equal(l[[name]], l.copy[[name]])
    }
  }
})


test_that("Multi-layer feedforward works", {
  net = network$new(
    x = matrix(rnorm(100), nrow = 20, ncol = 5),
    layers = list(
      l1 = layer$new(
        biases = rnorm(6),
        coefficients = matrix(rnorm(30), nrow = 5),
        nonlinearity = rectify,
        dim = c(5L, 6L),
        dropout = FALSE
      ),
      l2 = layer$new(
        biases = rnorm(7),
        coefficients = matrix(rnorm(42), nrow = 6),
        nonlinearity = sigmoid,
        dropout = FALSE
      )
    ),
    n.layers = 2L,
    minibatch.size = 5L
  )
  
  
  net$newMinibatch()
  net$feedForward()
  
  expect_equal(
    with(
      net$layers[[1]],
      nonlinearity((input %*% coefficients) %plus% biases)
    ),
    net$layers[[1]]$output
  )
  
  expect_equal(
    net$layers[[1]]$output,
    net$layers[[2]]$input
  )
  
  expect_equal(
    with(
      net$layers[[2]],
      nonlinearity((input %*% coefficients) %plus% biases)
    ),
    net$layers[[2]]$output
  )
})
