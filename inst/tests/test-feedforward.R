context("Feedforward")

test_that("Single-layer feedforward works", {
  l = createLayer(
    n.inputs = 4L,
    n.outputs = 7L,
    prior = gaussian.prior$new(mean = 0, var = 1),
    nonlinearity.name = "sigmoid",
    minibatch.size = 5L,
    n.importance.samples = 3L,
    updater.name = "sgd",
    updater.arguments = NULL
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
    l$nonlinearity$f((l$inputs[ , , 2] %*% l$coefficients) %plus% l$biases)
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
  net = mistnet(
    x = matrix(rnorm(100), nrow = 20, ncol = 5),
    y = matrix(rnorm(100), nrow = 20, ncol = 5),
    nonlinearity.names = c("rectify", "rectify", "sigmoid"),
    hidden.dims = c(13L, 17L),
    priors = list(
      gaussian.prior(mean = 0, var = .001),
      gaussian.prior(mean = 0, var = .001),
      gaussian.prior(mean = 0, var = .001)
    ),
    loss = bernoulliLoss(),
    minibatch.size = 4L,
    n.importance.samples = 27L,
    n.ranef = 3L,
    ranefSample = gaussianRanefSample,
    training.iterations = 0L
  )
  
  ranefs = net$ranefSample(nrow = net$minibatch.size, ncol = net$n.ranef)
  net$selectMinibatch()
  net$feedForward(
    cbind(
      net$x[net$minibatch.ids, ], 
      ranefs
    ),
    2
  )
  
  expect_equal(
    net$layers[[1]]$nonlinearity$f(
      (cbind(ranefs, net$x[net$minibatch.ids, ]) %*% net$layers[[1]]$coefficients) %plus% net$layers[[1]]$biases
    ),
    net$layers[[1]]$outputs[,,2]
  )
  
  expect_equal(
    net$layers[[1]]$outputs,
    net$layers[[2]]$inputs
  )
  
  expect_equal(
    with(
      net$layers[[2]],
      nonlinearity$f((inputs[,,2] %*% coefficients) %plus% biases)
    ),
    net$layers[[2]]$outputs[,,2]
  )
  
  
  expect_equal(
    with(
      net$layers[[3]],
      nonlinearity$f((inputs[,,2] %*% coefficients) %plus% biases)
    ),
    net$layers[[3]]$outputs[,,2]
  )
})
