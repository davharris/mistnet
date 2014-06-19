context("Prediction")

test_that("Prediction works",{
  
  x = dropoutMask(17L, 37L)
  y = dropoutMask(17L, 19L)
  n.importance.samples = 11L # Number for testing, not for training
  
  
  net = mistnet(
    x,
    y,
    layer.definitions = list(
      defineLayer(
        nonlinearity = rectify.nonlinearity(), 
        size = 23, 
        prior = gaussianPrior(0, 001)
      ),
      defineLayer(
        nonlinearity = rectify.nonlinearity(), 
        size = 31, 
        prior = gaussianPrior(0, 001)
      ),
      defineLayer(
        nonlinearity = sigmoid.nonlinearity(), 
        size = ncol(y), 
        prior = gaussianPrior(0, 001)
      )
    ),
    n.ranef = 3L,
    ranefSample = gaussianRanefSample,
    n.importance.samples = 10L,
    n.minibatch = nrow(x),
    training.iterations = 0L,
    loss = bernoulliLoss()
  )
  net$fit(1) # Feed forward
  net$layers[[3]]$biases[] = 0 # Undo bias updates
  
  p = predict(net, rbind(x, x), n.importance.samples = n.importance.samples)
  
  expect_equal(dim(p), c(2 * nrow(x), ncol(y), n.importance.samples))
  
  expect_true(all(p == 0.5)) # With no coefficients, everything should be 0.5
  
  # With these settings, prediction should just make a copy (except for latent
  # variables)
  copy = predict(
    net, 
    x, 
    n.importance.samples = net$n.importance.samples, 
    return.model = TRUE
  )
  
  copy2 = net$copy()
  
  
  # Everything but layers should be identical
  for(name in names(network$fields())){
    if(name != "layers"){
      expect_identical(copy$field(name), copy2$field(name))
    }
  }
  
})
