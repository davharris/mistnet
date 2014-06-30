#' @export
gaussianSampler = function(ncol, sd = 1){
  ncol = safe.as.integer(ncol)
  function(nrow){
    out = rnorm(nrow * ncol, sd = sd)
    dim(out) = c(nrow, ncol)
    
    out
  }
}
