context("Feedforward")

test_that("Single-layer feedforward works", {
  l = createLayer(
    n.inputs = 4L,
    n.outputs = 7L,
    n.minibatch = 5L,
    n.importance.samples = 3L,
    nonlinearity = sigmoid.nonlinearity(),
    prior = gaussian.prior(mean = 0, var = 1),
    updater = sgd.updater(momentum = .9, learning.rate = .001)
  )
  l.copy = l$copy()
  
  input.matrix = matrix(rnorm(20), ncol = 4)
  expect_error(l$forwardPass(input.matrix), "sample.num is missing")
  
  l$forwardPass(input.matrix, 2L)
  
  
  expect_equal(
    l$inputs[ , , 2],
    (input.matrix %*% l$coefficients) %plus% l$biases
  )
  expect_equal(
    l$outputs[ , , 2],
    l$nonlinearity$f((input.matrix %*% l$coefficients) %plus% l$biases)
  )
  
  # Nothing should change during feedforward except the listed fields
  for(name in layer$fields()){
    name.shouldnt.change = name %in% c("activations", "outputs")
    if(name.shouldnt.change){
    }else{
      expect_equal(l[[name]], l.copy[[name]])
    }
  }
})


test_that("Multi-layer feedforward works", {
  y = matrix(rnorm(100), nrow = 20, ncol = 5)
  net = mistnet(
    x = matrix(rnorm(100), nrow = 20, ncol = 5),
    y = y,
    layer.definitions = list(
      defineLayer(
        nonlinearity = rectify.nonlinearity(), 
        size = 23, 
        prior = gaussian.prior(mean = 0, var =  0.001)
      ),
      defineLayer(
        nonlinearity = rectify.nonlinearity(), 
        size = 31, 
        prior = gaussian.prior(mean = 0, var =  0.001)
      ),
      defineLayer(
        nonlinearity = sigmoid.nonlinearity(), 
        size = ncol(y), 
        prior = gaussian.prior(mean = 0, var =  0.001)
      )
    ),
    loss = bernoulliLoss(),
    n.minibatch = 4L,
    n.importance.samples = 27L,
    sampler = gaussianSampler(ncol = 3L),
    training.iterations = 0L
  )
  
  ranefs = net$sampler(nrow = net$n.minibatch)
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
    net$layers[[2]]$nonlinearity$f(
      (net$layers[[1]]$outputs[,,2] %*% net$layers[[2]]$coefficients) %plus% net$layers[[2]]$biases
    ),
    net$layers[[2]]$outputs[,,2]
  )
  
  
  expect_equal(
    net$layers[[3]]$nonlinearity$f(
      (net$layers[[2]]$outputs[,,2] %*% net$layers[[3]]$coefficients) %plus% net$layers[[3]]$biases
    ),
    net$layers[[3]]$outputs[,,2]
  )
})

