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
    l1.decay = "numeric",
    lr.multiplier = "numeric"
  ),
  contains = "nonlinearity",
  methods = list(
    f = mrf_meanfield
  )
)
