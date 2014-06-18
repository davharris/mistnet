setOldClass("prior")

gaussianPrior = function(mean, var){
  structure(
    list(
      family = "gaussian",
      getLogGrad = function(x){
        - (x - mean) / var
      },
      sample = function(n){
        rnorm(n, mean = mean, sd = sqrt(var))
      }
    ),
    class = "prior"
  )
}

laplacePrior = function(location, scale){
 structure(
   list(
     family = "laplace",
     getLogGrad = function(x){
       # This will work if the learning rate is small. Otherwise, it could 
       # overshoot past zero. That's probably not a big deal in practice?
       - sign(x - location) / scale
     },
     sample = function(n){
       rexp(n, rate  = 1 / scale) * sample(c(-1, 1), size = n, replace = TRUE)
     }
   ),
   class = "prior"
  ) 
}
