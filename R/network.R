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
    importance.weights = "matrix",
    loss = "function",
    lossGradient = "function",
    ranefSample = "function",
    n.ranef = "integer"
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
    feedForward = function(input, sample.num){
      # First layer gets the specified inputs
      layers[[1]]$forwardPass(input, sample.num)
      # Subsequent layers get their inputs from the layer preceding them
      if(n.layers > 1){
        for(j in 2:n.layers){
          layers[[j]]$forwardPass(layers[[j - 1]]$output, sample.num)
        }
      }
    },
    backprop = function(sample.num){
      # Final layer gets its error from the loss gradient
      net.output = layers[[n.layers]]$outputs[ , , sample.num]
      layers[[n.layers]]$backwardPass(
        lossGradient(y = y[minibatch.ids, ], yhat = net.output),
        sample.num
      )
      
      # Earlier layers' error gradients are filtered through the coefficients of
      # the layer above.
      if(n.layers > 1){
        for(i in (n.layers - 1):1){
          layers[[i]]$backwardPass(
            tcrossprod(
              layers[[i + 1]]$error.grads[ , , sample.num], 
              layers[[i + 1]]$coefficients
            )
          )
        }
      }
    },
    updateCoefficients = function(){
      for(i in 1:n.layers){
        layers[[i]]$updateCoefficients()
      }
    },
    fit = function(iterations){
      for(i in 1:iterations){
        newMinibatch()
        estimateGradient()
        updateCoefficients()
      }
    },
    estimateGradient = function(){
      for(i in 1:n.importance.samples){
        feedForward(
          cbind(
            x[minibatch.ids, ], 
            ranefSample(nrow = minibatch.size, ncol = n.ranef)
          ),
          i
        )
        backprop(i)
      }
      averageSampleGradients()
    },
    findImportanceWeights = function(){
      for(i in 1:n.importance.samples){
        importance.errors[ , i] = rowSums(
          yhat = layers[[n.layers]]$outputs[ , , i],
          loss(y = y[minibatch.ids, ], yhat = yhat)
        )
      }
      unscaled.weights = t(apply(
        importance.errors, 
        1,
        function(x) exp(min(x) - x)
      ))
      importance.weights <<- unscaled.weights / rowSums(unscaled.weights)
    },
    averageSampleGradients = function(){
      findImportanceWeights()
      
      for(i in 1:n.layers){
        with(
          layers[[i]],{
            weighted.bias.grads = 0 * weighted.bias.grads
            weighted.llik.grads = 0 * weighted.llik.grads
            for(i in 1:n.importance.samples){
              w = weights[ , i]
              weighted.bias.grads = bias.grad + w * bias.grads[ , i]
              weighted.llik.grads = llik.grad + w * llik.grads[ , , i]
            }
          }
        )
      }
    }
  )
)
