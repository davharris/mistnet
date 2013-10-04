network = setRefClass(
  Class = "network",
  fields = list(
    x = "matrix",
    y = "matrix",
    layers = "list",
    n.layers = "integer",
    dataset.size = "integer",
    minibatch.size = "integer",
    minibatch.ids = "integer",
    n.importance.samples = "integer",
    importance.weights = "matrix",
    loss = "function",
    lossGradient = "function",
    ranefSample = "function",
    n.ranef = "integer",
    learning.rate = "numeric",
    momentum = "numeric",
    completed.iterations = "integer"
  ),
  methods = list(
    
    fit = function(iterations){
      if(iterations < 1L){
        if(iterations == 0L){
          return(NULL)
        }else{
          stop(paste0(iterations, " is not a valid number of iterations"))
        }
      }
      # Maybe put some (optional) assertions here?
      # Do I have an opinion about non-integer iteration counts?
      for(i in 1:iterations){
        selectMinibatch()
        estimateGradient()
        updateCoefficients()
        completed.iterations <<- completed.iterations + 1L
      }
    },
    
    selectMinibatch = function(row.nums){
      if(missing(row.nums)){
        minibatch.ids <<- sample.int(nrow(x), minibatch.size, replace = FALSE)
      }else{
        if(length(row.nums) != minibatch.size){
          minibatch.size <<- length(row.nums)
          for(i in 1:n.layers){
            layers[[i]]$resetState(
              minibatch.size = minibatch.size, 
              n.importance.samples
            )
          }
        }
        minibatch.ids <<- row.nums
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
    
    updateCoefficients = function(){
      for(i in 1:n.layers){
        layers[[i]]$updateCoefficients(
          learning.rate = learning.rate, 
          momentum = momentum,
          dataset.size = dataset.size
        )
      }
    },
    
    feedForward = function(input, sample.num){
      # First layer gets the specified inputs
      layers[[1]]$forwardPass(input, sample.num)
      # Subsequent layers get their inputs from the layer preceding them
      if(n.layers > 1){
        for(j in 2:n.layers){
          layers[[j]]$forwardPass(
            layers[[j - 1]]$outputs[ , , sample.num], 
            sample.num
          )
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
          incoming.error.grad = layers[[i]]$backwardPass(
            tcrossprod(
              layers[[i + 1]]$error.grads[ , , sample.num], 
              layers[[i + 1]]$coefficients
            ),
            sample.num = sample.num
          )
        }
      }
    },
    
    averageSampleGradients = function(){
      findImportanceWeights()
      for(i in 1:n.layers){
        layers[[i]]$combineSampleGradients(
          weights = importance.weights,     
          n.importance.samples = n.importance.samples
        )
      }
    },
    
    findImportanceWeights = function(){
      importance.errors = zeros(minibatch.size, n.importance.samples)
      for(i in 1:n.importance.samples){
        importance.errors[ , i] = rowSums(
          loss(
            y = y[minibatch.ids, ], 
            yhat = layers[[n.layers]]$outputs[ , , i]
          )
        )
      }
      importance.weights <<- weighImportance(importance.errors)
    },
    
    
    predict = function(newdata, n.importance.samples = 1000L, return.model.copy = FALSE){
      copy = .self$copy()
      copy$n.importance.samples = n.importance.samples
      # ensure that selectMinibatch triggers its extra routines for changing
      # the number of importance samples.
      # In retrospect, this is a screwy way to control things.
      copy$selectMinibatch(1L)
      copy$selectMinibatch(1:nrow(newdata))
      
      # Create an input matrix with the correct dimensions & correct values
      # everywhere that's fixed.
      inputs = cbind(
        newdata, 
        zeros(nrow = nrow(newdata), ncol = n.ranef)
      )
      
      
      for(i in 1:copy$n.importance.samples){
        inputs[, -(1:ncol(newdata))] = copy$ranefSample(
          nrow = nrow(newdata), 
          ncol = n.ranef
        )
        copy$feedForward(
          inputs,
          i
        )
      }
      
      if(return.model.copy){
        return(copy)
      }else{
        return(copy$layers[[length(copy$layers)]]$outputs)
      }
    }
  )
)
