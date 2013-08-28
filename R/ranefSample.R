gaussianRanefSample = function(nrow, ncol){
  out = rnorm(nrow * ncol)
  dim(out) = c(nrow, ncol)
  
  out
}
