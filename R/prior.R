#' @exportClass prior
prior = setRefClass(
  Class = "prior",
  fields = list(),
  methods = list(
    getLogGrad = function(x) stop({"log gradient not defined for this prior"}),
    sample = function(n){stop("sampler not defined for this prior")}
  )
)


#' @export
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
    },
    sample = function(n){
      rnorm(n, mean = mean, sd = sqrt(var))
    }
  )
)

#' @export
laplacePrior = setRefClass(
  Class = "laplacePrior",
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
