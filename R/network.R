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
    lossGradient = "function"
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
        newMinibatch()
        for(j in 1:n.importance.samples){
          if(n.importance.samples > 1){sampleXFromPrior()}
          feedForward()
          backprop()
          if(n.importance.samples > 1){saveGradients(j)}
        }
        if(n.importance.samples > 1){
          averageSampleGradients()
          resetSavedGradients()
        }
        updateCoefficients()
      }
    },
    sampleXFromPrior = function(){
      # Do nothing
    },
    averageSampleGradients = function(){
      # Do nothing
    },
    resetSavedGradients = function(){
      # Do nothing
    },
    reportLoss = function(){
      loss(y = y[minibatch.ids, ], yhat = layers[[n.layers]]$output)
    }
  )
)
