# Activation functions ----------------------------------------------------

sigmoid = function(x){
  # Sometimes slightly slower than plogis, but has a floor at 2.220446e-16 to
  # prevent division-by-zero errors.
  storage.mode(x) = "numeric"
  make.link("logit")$linkinv(x)
}

linear = function(x){
  x
}

# Note: rectify is defined in src/rectify.cpp
#       Also, exp is just base::exp

# Gradients ---------------------------------------------------------------


sigmoidGrad = function(x){
  make.link("logit")$mu.eta(x)
}

rectifyGrad = function(x){
  x > 0
}

linearGrad = function(x){
  out = rep(1, length(x))
  dim(out) = dim(x)
  out
}

expGrad = function(x){
  exp(x)
}


# Nonlinearity classes ----------------------------------------------------

#' @include updater.R
#' @export nonlinearity
#' @exportClass nonlinearity
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

#' @exportClass linear.nonlinearity
#' @export linear.nonlinearity
linear.nonlinearity = setRefClass(
  Class = "linear.nonlinearity",
  fields = list(),
  contains = "nonlinearity",
  methods = list(
    f = linear,
    grad = linearGrad
  )
)

#' @exportClass sigmoid.nonlinearity
#' @export sigmoid.nonlinearity
sigmoid.nonlinearity = setRefClass(
  Class = "sigmoid.nonlinearity",
  fields = list(),
  contains = "nonlinearity",
  methods = list(
    f = sigmoid,
    grad = sigmoidGrad
  )
)

#' @exportClass rectify.nonlinearity
#' @export rectify.nonlinearity
rectify.nonlinearity = setRefClass(
  Class = "rectify.nonlinearity",
  fields = list(),
  contains = "nonlinearity",
  methods = list(
    f = rectify,
    grad = rectifyGrad
  )
)


#' @exportClass exp.nonlinearity
#' @export exp.nonlinearity
exp.nonlinearity = setRefClass(
  Class = "exp.nonlinearity",
  fields = list(),
  contains = "nonlinearity",
  methods = list(
    f = function(x){
      # Wrapping exp(x) in a function is apparently necessary becuase exp() is
      # primitive and thus not a closure.  There does not seem to be a speed
      # penalty for this wrapper.
      exp(x)
    },
    grad = expGrad
  )
)



#' @exportClass mf_mrf.nonlinearity
#' @export mf_mrf.nonlinearity
mf_mrf.nonlinearity = setRefClass(
  Class = "mf_mrf.nonlinearity",
  fields = list(
    lateral = "matrix",
    maxit = "integer",
    damp = "numeric",
    tol = "numeric",
    l1.decay = "numeric",
    updater = "updater"
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
      importance.weights,
      n.importance.samples,
      dataset.size
    ){
      observed.crossprod = crossprod(observed)
      
      predicted.crossprod = findPredictedCrossprod(
        predicted, 
        importance.weights
      )
      
      diff = predicted.crossprod - observed.crossprod
      penalty = sign(lateral) * l1.decay
      
      updater$computeDelta((diff / nrow(observed) + penalty))
      diag(updater$delta) <<- 0
      
      lateral <<- lateral + updater$delta
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
    # The square root is there because things get multiplied together in
    # the cross product. Sqrt undoes this multiplication.
    crossprod.increment = crossprod(
      predicted[ , , i] * sqrt(importance.weights[ , i])
    )
    predicted.crossprod = predicted.crossprod + crossprod.increment
  }
  
  predicted.crossprod
}
