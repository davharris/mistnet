#' @exportClass prior
prior = setRefClass(
  Class = "prior",
  fields = list(),
  methods = list(
    getLogGrad = function(x) stop("log gradient not defined for this prior")
  )
)


gaussian.prior = setRefClass(
  Class = "gaussian.prior",
  fields = list(
    mean = "numeric",
    var = "numeric"
  ),
  contains = "prior",
  methods = list(
    getLogGrad = function(x){
      - (x - .self$mean) / .self$var
    }
  )
)

laplace.prior = setRefClass(
  Class = "laplace.prior",
  fields = list(
    location = "numeric",
    scale = "numeric"
  ),
  contains = "prior",
  methods = list(
    getLogGrad = function(x){
      # This will work if the learning rate is small. Otherwise, it could 
      # overshoot. That's probably not a big deal, though...
      - sign(x - .self$location) / .self$scale
    }
  )
)

# Why a Student-t distribution with 3 degrees of fredom?
# Because it's the one with the most (non-infinite) degrees of freedom that 
# Wikipedia has a simple formula for. I don't want to deal with derivatives
# of gamma functions, so I'm not sure if I'll make a more general t prior.
t3.prior = setRefClass(
  Class = "t3.prior",
  fields = list(
    location = "numeric",
    scale = "numeric"
  ),
  contains = "prior",
  methods = list(
    getLogGrad = function(x){
      z = (x - location) / scale
      - 4 * x / (x^2 + 3)
    }
  )
)

flat.prior = setRefClass(
  Class = "flat.prior",
  fields = list(),
  contains = "prior",
  methods = list(
    getLogGrad = function(x){
      0
    }
  )
)