network$methods(
  update_all = function(n.steps){
    # Update optimization hyperparameters
    .self$momentum = min((1 + .self$completed.iterations / 1000) / 2, .99)
    
    # Dividing at the end to undo the 80% initial momentum
    .self$learning.rate = starting.rate / 
      (1 + 1E-4 * .self$completed.iterations) * (1 - .self$momentum) / .8
    
    # Hack to keep rectified units alive: add a small amount to biases of "dead"
    # units whose activations are always negative
    for(layer.num in 1:.self$n.layers){
      if(identical(.self$layers[[layer.num]]$nonlinearity$f,rectify)){
        for(hidden.num in 1:.self$layers[[layer.num]]$coef.dim[[2]]){
          if(max(.self$layers[[layer.num]]$activations[,hidden.num,]) < 0){
            .self$layers[[layer.num]]$biases[[hidden.num]] = 
              .self$layers[[layer.num]]$biases[[hidden.num]] + .01
          }
        }
      }
    }
    
    # Update priors
    if(.self$completed.iterations > 100){
      for(layer.num in 1:.self$n.layers){
        # Update prior variance of all layers
        .self$layers[[layer.num]]$prior$var = apply(
          .self$layers[[layer.num]]$coefficients, 
          1, 
          var
        )
        #  (prior variance has a lower bound)
        .self$layers[[layer.num]]$prior$var = pmax(
          .self$layers[[layer.num]]$prior$var, 
          .001
        )
        if(layer.num == .self$n.layers){
          dim = .self$layers[[layer.num]]$coef.dim[2]
          # Update prior mean of last layer.  Pull it in sligthly from the
          # observed mean, as if there were one observation at exactly 0.
          .self$layers[[layer.num]]$prior$mean = rowMeans(
            .self$layers[[layer.num]]$coefficients
          ) * dim / (dim + 1)
        }
        # Variances are drawn toward a common value.
        var = .self$layers[[layer.num]]$prior$var
        newvar = (var * length(var) + mean(var)) / (length(var)  + 1)
        .self$layers[[layer.num]]$prior$var = newvar
      }
    }
    
    .self$fit(n.steps)
  }
)


# Warning: This function grabs hyperparameters from the global environment
# for some reason.
buildNet = function(x, y){
  order = sample.int(nrow(x))
  net = mistnet(
    x = x[order, ],
    y = y[order, ],
    nonlinearity.names = c("rectify", "linear", "sigmoid"),
    hidden.dims = c(n.layer1, n.layer2),
    priors = list(
      gaussian.prior(mean = 0, var = 1),
      gaussian.prior(mean = 0, var = 1),
      gaussian.prior(mean = 0, var = 1)
    ),
    learning.rate = starting.rate,
    momentum = .8,
    loss = crossEntropy,
    lossGrad = crossEntropyGrad,
    minibatch.size = minibatch.size,
    n.importance.samples = n.importance.samples,
    n.ranef = n.ranef,
    ranefSample = gaussianRanefSample,
    training.iterations = 0L
  )
  
  # Initialize coefficients and biases
  net$layers[[3]]$biases = qlogis(colMeans(y))
  net$layers[[1]]$coefficients[,] = rt(prod(net$layers[[1]]$coef.dim), df = 5) / 5
  net$layers[[2]]$coefficients[,] = rt(prod(net$layers[[2]]$coef.dim), df = 5) / 5
  net$layers[[3]]$coefficients[,] = rt(prod(net$layers[[3]]$coef.dim), df = 5) / 5
  
  # Fill things in so fitting doesn't break on the very first iteration
  net$selectMinibatch()
  net$estimateGradient()
  
  net
}

cv.evaluate = function(){
  prediction.array = predict(
    net,
    scale(env)[in.train, ][!in.fold, ], 
    n.prediction.samples
  )
  
  cv.losses = apply(
    prediction.array,
    3,
    function(x){
      rowSums(
        net$loss(
          y = route.presence.absence[in.train, ][!in.fold, ], 
          yhat = x
        )
      )
    }
  )
  
  mean(findLogExpectedLik(cv.losses))
}
