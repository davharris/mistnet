#' Network
#'
#' @description A reference class object for a mistnet \code{network} object.
#'
#' @details __
#'
#' @field x a numeric matrix of predictor variables.  One row
#'  per observation, one column per predictive feature.
#' @field y a \code{matrix} of responses to \code{x}.  One row per observation, one
#'  column per response variable.
#' @field layers a \code{list} of \code{\link{layer}} objects
#' @field n.layers an integer corresponding to \code{length(layers)}
#' @field dataset.size an integer corresponding to the number of rows in 
#'  \code{x} and \code{y}
#' @field n.minibacth an \code{integer} specifying the number of rows to include
#'  in each stochastic estimate of the likelihood gradient.
#' @field n.importance.samples an \code{integer}
#' @field importance.weights a numeric matrix containing the weights associated
#'  with the most recent round of importance sampling.  (one row per observation,
#'  one column per Monte Carlo sample).
#' @field loss the a \code{\link{loss}} object
#' @field sampler the function used to generate Monte Carlo samples for 
#'  importance sampling
#' @field completed.iterations a counter that increments after each iteration
#'  of model fitting
#' @field debug a logical flag indicating whether special debugging measures
#'  should be enabled. Useful for diagnosing problems with the model, but 
#'  potentially slow.
#' @export network
#' @exportClass network
#' @seealso \code{\link{mistnet}}, \code{\link{layer}}
#' @include row.selector.R
#' @include sampler.R
network = setRefClass(
  Class = "network",
  fields = list(
    x = "matrix",
    y = "matrix",
    inputs = "array",
    layers = "list",
    n.layers = "integer",
    dataset.size = "integer",
    row.selector = "row.selector",
    n.importance.samples = "integer",
    importance.weights = "matrix",
    loss = "loss",
    sampler = "sampler",
    completed.iterations = "integer",
    debug = "logical"
  ),
  methods = list(

    fit = function(iterations){
      "Update the coefficients of the network object iteratively by minibatch gradient 
       descent. In each iteration, the gradient is estimated using importance sampling,
       as described in (Tang and Salakhutdinov, ICML 2013) and in (Harris, Methods in 
       Ecology and Evolution 2015)."
      
      if(iterations < 1L){
        if(iterations == 0L){
          return(NULL)
        }else{
          stop(paste0(iterations, " is not a valid number of iterations"))
        }
      }

      for(i in 1:iterations){
        selectMinibatch()    # Identify the rows to use in this iteration
        estimateGrad()       # Feedforward, then backprop for approximate gradients
        updateCoefficients()
        completed.iterations <<- completed.iterations + 1L
        
        if(debug){
          for(layer in layers){
            assert_that(!any(is.na(layer$weights)))
            assert_that(!any(is.na(layer$outputs)))
          }
        }
      }
    },
    
    selectMinibatch = function(row.nums){
      "Select which rows to use in this iteration, or re-initialize the row.selector"
      
      if(missing(row.nums)){
        row.selector$select()
      }else{
        # Re-initialize the row.selector with the new size
        n = length(row.nums)
        row.selector$n.minibatch <<- n
        for(i in 1:n.layers){
          layers[[i]]$resetState(
            n.minibatch = row.selector$n.minibatch, 
            n.importance.samples
          )
        }
        row.selector$minibatch.ids <<- row.nums
      }
    },
    
    estimateGrad = function(){
      "Feedforward once per Monte Carlo sample, then calculate the associated
      gradients by backpropagation.  Finally, average the gradients according
      to the importance weight of each sample."
      
      for(i in 1:n.importance.samples){
        # Concatenate the relevant rows of x with random Monte Carlo samples
        inputs[,,i] <<- cbind(
          x[row.selector$minibatch.ids, ], 
          sampler$sample(nrow = row.selector$n.minibatch)
        )
        feedForward(
          inputs[,,i],
          i
        )
        backprop(i)
      }
      averageSampleGrads()
    },
    
    updateCoefficients = function(){
      "Update the weights, biases, and (possibly) other coefficients of
       each layer in the network (if they exist)."
      
      for(i in 1:n.layers){
        # Update the weights and biases in each layer
        layers[[i]]$updateCoefficients(
          dataset.size = dataset.size,
          n.minibatch = row.selector$n.minibatch
        )
        
        # Some special layers have additional coefficients to update
        layers[[i]]$nonlinearity$update(
          observed = y[row.selector$minibatch.ids, ],
          predicted = layers[[i]]$outputs, 
          importance.weights = importance.weights,
          dataset.size = dataset.size
        )
      }
    },
    
    feedForward = function(input, sample.num){
      "Use the network to generate one Monte Carlo prediction of y given x."
      
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
      "Backpropagate the errors of one Mone Carlo sample through the network "
      
      # Final layer gets its error from the loss gradient
      layers[[n.layers]]$backwardPass(
        loss$grad(
          y = y[row.selector$minibatch.ids, ], 
          yhat = layers[[n.layers]]$outputs[ , , sample.num]
        ),
        sample.num
      )
      
      # Earlier layers' error gradients are filtered through the weights of
      # the layer above.
      if(n.layers > 1){
        for(i in (n.layers - 1):1){
          incoming.error.grad = layers[[i]]$backwardPass(
            tcrossprod(
              layers[[i + 1]]$error.grads[ , , sample.num], 
              layers[[i + 1]]$weights
            ),
            sample.num = sample.num
          )
        }
      }
    },
    
    averageSampleGrads = function(){
      "Combine the gradients found in different Monte Carlo samples according
       to their importance weights."
      
      findImportanceWeights()
      for(i in 1:n.layers){
        layers[[i]]$combineSampleGrads(
          # The first layer gets data from the network inputs; subsequent
          # layers get inputs from the previous layer.
          inputs = if(i==1){inputs}else{layers[[i - 1]]$outputs},
          weights = importance.weights,     
          n.importance.samples = n.importance.samples
        )
      }
    },
    
    findImportanceWeights = function(){
      "Calculate the importance weights for each Monte Carlo sample for each row
       of the minibatch."
      
      importance.errors = zeros(row.selector$n.minibatch, n.importance.samples)
      for(i in 1:n.importance.samples){
        importance.errors[ , i] = rowSums(
          loss$loss(
            y = y[row.selector$minibatch.ids, ], 
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
