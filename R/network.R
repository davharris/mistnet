# documentation should note that ranefSample should have mean zero.

network = setRefClass(
  Class = "network",
  fields = list(
    x = "matrix",
    y = "matrix",
    layers = "list",
    n.layers = "integer",
    minibatch.size = "integer",
    minibatch.ids = "integer",
    n.importance.samples = "integer",
    loss = "function",
    lossGradient = "function",
    ranefSample = "function",
    n.ranef = "integer",
    importance.errors = "numeric"  
  ),
  methods = list(
    newMinibatch = function(row.nums){
      if(missing(row.nums)){
        minibatch.ids <<- sample.int(nrow(x), minibatch.size, replace = FALSE)
      }else{
        # Should this check that length(row.nums) == minibatch.size?
        minibatch.ids <<- row.nums
      }
    },
    feedForward = function(inputs){
      if(missing(inputs)){inputs = x[minibatch.ids, ]}
      
      # First layer gets its inputs from x
      layers[[1]]$forwardPass(inputs)
      
      # Subsequent layers get their inputs from previous layers
      if(n.layers > 1){
        for(i in 2:n.layers){
          layers[[i]]$forwardPass(layers[[i - 1]]$output)
        }
      }
    },
    backprop = function(){
      
      # Final layer just sees error from the loss gradient
      layers[[n.layers]]$backwardPass(
        lossGradient(y = y[minibatch.ids, ], yhat = layers[[n.layers]]$output)
      )
      
      # Earlier layers' error gradients are filtered through the coefficients of
      # the layer above.
      if(n.layers > 1){
        for(i in (n.layers - 1):1){
          layers[[i]]$backwardPass(
            tcrossprod(
              layers[[i + 1]]$error.grad, 
              layers[[i + 1]]$coefficients
            )
          )
        }
      }
    },
    updateCoefficients = function(){
      for(layer in layers){
        layer$updateCoefficients()
      }
    },
    predict = function(newdata){
      feedForward(newdata)
      return(output)
    },
    fit = function(iterations){
      for(i in 1:iterations){
        
        # Step 1: pick a minibatch.
        newMinibatch()
        
        # Step 2: Find gradients on that minibatch.
        if(n.importance.samples == 1L){
          feedForward()
          backprop()
        }else{
          findImportanceSampleGradients()
        }
        
        # Step 3: Update coefficients.
        updateCoefficients()
      }
    },
    findImportanceSampleGradients = function(){
      for(j in 1:n.importance.samples){
        feedForward(
          cbind(
            x[minibatch.ids, ], 
            ranefSample(nrow = minibatch.size, ncol = n.ranef)
          )
        )
        backprop()
        saveGradients(j)
        saveImportanceError(j)
      }
      
      averageSampleGradients()
      resetImportanceSampler()
    },
    saveGradients = function(sample.number){
      for(layer in layers){
        layer$importance.bias.grads[ , sample.number] = layer$bias.grad
        layer$importance.llik.grads[ , , sample.number] = layer$llik.grad
      }
    },
    averageSampleGradients = function(){
      unscaled.weights = exp(min(importance.errors) - importance.errors)
      weights = unscaled.weights / sum(unscaled.weights)
      
      layer$bias.grad = 0
      layer$llik.grad = 0
      
      for(layer in layers){
        for(i in 1:n.importance.samples){
          layer$bias.grad = layer$bias.grad + weights[i] * bias.grads[ , iter]
          layer$llik.grad = layer$llik.grad + weights[i] * llik.grads[ , , iter]
        }
      }
    },
    resetImportanceSampler = function(){
      "If everything is working properly, this shouldn't be necessary. But this
      step is still useful because it might prevent the code from failing
      silently."
      
      # TODO
    },
    saveImportanceError = function(sample.number){
      importance.errors[sample.number] <<- sum(reportLoss())
    },
    reportLoss = function(){
      loss(y = y[minibatch.ids, ], yhat = layers[[n.layers]]$output)
    }
  )
)
