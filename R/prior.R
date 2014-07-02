#' @exportClass prior
prior = setRefClass(
  Class = "prior",
  fields = list(),
  methods = list(
    getLogGrad = function(x) stop({"log gradient not defined for this prior"}),
    sample = function(n){stop("sampler not defined for this prior")},
    update = function(){stop("update not defined for this prior")}
  )
)

# Annoyingly, var's class will be different if it's a scalar or a matrix...
#  Set to "any" as a stopgap.
#' @exportClass gaussian.prior
gaussian.prior = setRefClass(
  Class = "gaussian.prior",
  fields = list(
    mean = "numeric",
    var = "ANY"
  ),
  contains = "prior",
  methods = list(
    getLogGrad = function(x){
      - (x - .self$mean) / .self$var
    },
    sample = function(n){
      rnorm(n, mean = mean, sd = sqrt(var))
    },
    update = function(coefficients, update.mean, update.var, min.var){
      if(update.mean){
        mean <<- rowMeans(coefficients)
      }
      if(update.var){
        var.vector = apply(coefficients, 1, var)
        var.vector = pmax(
          (var.vector + mean(var.vector)) / 2,
          min.var
        )
        var <<- replicate(ncol(coefficients), var.vector)
      }
    }
  )
)

#' @exportClass laplace.prior
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
      # overshoot past zero. That's probably not a big deal in practice?
      - sign(x - location) / scale
    },
    sample = function(n){
      rexp(n, rate  = 1 / scale) * sample(c(-1, 1), size = n, replace = TRUE)
    }
  )
) 
