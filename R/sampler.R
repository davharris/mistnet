sampler = setRefClass(
  Class = "sampler",
  fields = list(),
  methods = list(
    sample = function(...){
      stop("no sampling method defined for this sampler")
    }
  )
)

#' @export gaussian.sampler
#' @exportClass gaussian.sampler
gaussian.sampler = setRefClass(
  Class = "gaussian.sampler",
  fields = list(
    ncol = "integer",
    sd = "numeric"
  ),
  methods = list(
    sample = function(nrow){
      out = rnorm(nrow * ncol, sd = sd)
      dim(out) = c(nrow, ncol)
      
      out
    }
  ),
  contains = "sampler"
)
