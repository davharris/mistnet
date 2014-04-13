# Activation functions ----------------------------------------------------

sigmoid = function(x){
  # Benchmarking suggests that this is 20% faster than plogis()  
  storage.mode(x) = "numeric"
  binomial()$linkinv(x)
}

linear = function(x){
  x
}

# Note: rectify is defined in src/rectify.cpp



# Gradients ---------------------------------------------------------------


sigmoidGrad = function(x){
  s = sigmoid(x)
  s * (1 - s)
}

rectifyGrad = function(x){
  x > 0
}

linearGrad = function(x){
  1
}


# Nonlinearity classes ----------------------------------------------------

nonlinearity = setRefClass(
  Class = "nonlinearity",
  fields = list(),
  methods = list(
    f = function(x){
      stop("activation function not defined for this nonlinearity")
    },
    grad = function(x){
      stop("gradient not defined for this nonlinearity")
    },
    update = function(...){
      # Do nothing
    }
  )
)

linear.nonlinearity = setRefClass(
  Class = "linear.nonlinearity",
  fields = list(),
  contains = "nonlinearity",
  methods = list(
    f = linear,
    grad = linearGrad
  )
)

sigmoid.nonlinearity = setRefClass(
  Class = "sigmoid.nonlinearity",
  fields = list(),
  contains = "nonlinearity",
  methods = list(
    f = sigmoid,
    grad = sigmoidGrad
  )
)


rectify.nonlinearity = setRefClass(
  Class = "rectify.nonlinearity",
  fields = list(),
  contains = "nonlinearity",
  methods = list(
    f = rectify,
    grad = rectifyGrad
  )
)


mf_mrf.nonlinearity = setRefClass(
  Class = "mf_mrf.nonlinearity",
  fields = list(
    lateral = "matrix",
    maxit = "integer",
    damp = "numeric",
    tol = "numeric",
    delta = "matrix",
    l1.decay = "numeric"
  ),
  contains = "nonlinearity",
  methods = list(
    f = function(x){
      mrf_meanfield(
        rinput = x, 
        rlateral = lateral, 
        maxit = maxit, 
        damp = damp, 
        tol = tol
      )
    },
    grad = sigmoidGrad,
    update = function(
      observed, 
      predicted, 
      learning.rate,
      importance.weights,
      n.importance.samples,
      momentum,
      dataset.size
    ){
      observed.crossprod = crossprod(observed)
      
      predicted.crossprod = findPredictedCrossprod(
        predicted, 
        importance.weights
      )
      
      diff = observed.crossprod - predicted.crossprod
      penalty = sign(lateral) * l1.decay
      

      scaled.learning.rate = learning.rate/ nrow(observed)
      delta <<- momentum * delta + scaled.learning.rate * (diff - penalty) 
      diag(delta) <<- 0
      
      lateral <<- lateral + delta
    }
  )
)

findPredictedCrossprod = function(predicted, importance.weights){
  predicted.crossprod = matrix(
    0, 
    ncol = dim(predicted)[[2]], 
    nrow = dim(predicted)[[2]]
  )
  for(i in 1:ncol(importance.weights)){
    # The square root is necessary because things get multiplied together in
    # the cross product.
    crossprod.increment = crossprod(
      predicted[ , , i] * sqrt(importance.weights[ , i])
    )
    predicted.crossprod = predicted.crossprod + crossprod.increment
  }
  
  predicted.crossprod
}
