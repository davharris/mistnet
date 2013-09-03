gaussianRanefSample = function(nrow, ncol, sd = 1){
  out = rnorm(nrow * ncol, sd = sd)
  dim(out) = c(nrow, ncol)
  
  out
}
