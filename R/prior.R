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

t.prior = setRefClass(
  Class = "t.prior",
  fields = list(
    location = "numeric",
    scale = "numeric",
    df = "numeric"
  ),
  contains = "prior",
  methods = list(
    getLogGrad = function(x){
      stop("t.prior gradient not yet defined")
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