#' Network
#'
#' @description A reference class object for a mistnet \code{network} object.
#'
#' @details __
#'
#' @field x a numeric matrix of predictor variables.  One row
#'  per example, one column per predictive feature.
#' @field y a \code{matrix} of responses to \code{x}.  One row per example, one
#'  column per response variable.
#' @field layers a \code{list} of \code{layer} objects
#' @field n.layers an integer corresponding to \code{length(layers)}
#' @field dataset.size an integer corresponding to the number of rows in 
#'  \code{x} and \code{y}
#' @field n.minibacth an \code{integer} specifying the number of rows to include
#'  in each stochastic estimate of the likelihood gradient.
#' @field minibatch.ids an \code{integer} vector specifying which rows of the 
#'  data set to include in the next estimate of the likelihood gradient
#' @field n.importance.samples an \code{integer}
#' @field importance.weights a numeric matrix containing the weights associated
#'  with the most recent round of importance sampling.  (one row per example,
#'  one column per Monte Carlo sample).
#' @field loss the loss function being optimized (not a \code{loss} object!)
#' @field lossGradient the gradient of the loss function being optimized
#' @field sampler the function used to generate Monte Carlo samples for 
#'  importance sampling
#' @field completed.iterations a counter that increments after each iteration
#'  of model fitting
#' @export
network = setRefClass(
  Class = "network",
  fields = list(
    x = "matrix",
    y = "matrix",
    layers = "list",
    n.layers = "integer",
    dataset.size = "integer",
    n.minibatch = "integer",
    minibatch.ids = "integer",
    n.importance.samples = "integer",
    importance.weights = "matrix",
    loss = "function",
    lossGradient = "function",
    sampler = "function",
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
    
    fitFinalNonlinearityOnly = function(iterations){
      for(iter in 1:iterations){
        
        selectMinibatch()
        
        for(i in 1:n.importance.samples){
          feedForward(
            cbind(
              x[minibatch.ids, ], 
              sampler(nrow = n.minibatch)
            ),
            i
          )
        }
        
        findImportanceWeights()
        
        layers[[n.layers]]$nonlinearity$update(
          observed = y[minibatch.ids, ],
          predicted = layers[[n.layers]]$outputs, 
          learning.rate = learning.rate / 10,
          importance.weights = importance.weights,
          momentum = momentum
        )
        
        # Do I want to include this?
        completed.iterations <<- completed.iterations + 1L
      }
    },
    
    selectMinibatch = function(row.nums){
      if(missing(row.nums)){
        stopifnot(n.minibatch > 0)
        start = (completed.iterations * n.minibatch) %% nrow(x)
        minibatch.ids <<- 1L + (seq(start, start + n.minibatch - 1) %% nrow(x))
      }else{
        if(length(row.nums) != n.minibatch){
          n.minibatch <<- length(row.nums)
          for(i in 1:n.layers){
            layers[[i]]$resetState(
              n.minibatch = n.minibatch, 
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
            sampler(nrow = n.minibatch)
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
          dataset.size = dataset.size,
          n.minibatch = n.minibatch
        )
        layers[[i]]$nonlinearity$update(
          observed = y[minibatch.ids, ],
          predicted = layers[[i]]$outputs, 
          importance.weights = importance.weights
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
      layers[[n.layers]]$backwardPass(
        lossGradient(
          y = y[minibatch.ids, ], 
          yhat = layers[[n.layers]]$outputs[ , , sample.num]
        ),
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
      importance.errors = zeros(n.minibatch, n.importance.samples)
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
    copy = function(shallow = FALSE){
      # Based on the default copy function provided for ReferenceClasses in 
      # the methods package.
      # A separate method is needed to ensure that the list of layers is copied
      # fully.  With the default method, the copied list contains the same
      # layer objects as the original.
      
      if(shallow){
        stop("Network objects can only be copied with shallow = FALSE")
      }
      
      original.layers = lapply(layers, function(x) x$copy())
      
      def = .refClassDef
      value = new("network")
      vEnv = as.environment(value)
      selfEnv = as.environment(.self)
      for (field in names(def@fieldClasses)) {
        current = get(field, envir = selfEnv)
        if(is(current, "envRefClass")){
          current = current$copy(shallow = FALSE)
        } 
        assign(field, current, envir = vEnv)
      }
      
      value$layers = original.layers
      
      value
    }
  )
)
