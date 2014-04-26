updater = setRefClass(
  Class = "updater",
  fields = list(
    delta = "matrix"
  ),
  methods = list(
    computeDelta = function(...){
      stop("computeDelta not defined for this updater")
    }
  )
)

momentum.updater = setRefClass(
  Class = "momentum.updater",
  contains = "updater",
  fields = list(
    momentum = "numeric",
    delta = "matrix"
  ),
  methods = list(
    computeDelta = function(gradient){
      delta <<- delta * momentum + gradient
    }
  )
)

adadelta.updater = setRefClass(
  Class = "adadelta.updater",
  contains = "updater",
  fields = list(
    rho = "numeric",
    epsilon = "numeric",
    squared.grad = "matrix",
    delta = "matrix",
    squared.delta = "matrix"
  ),
  methods = list(
    computeDelta = function(gradient){
      # Line numbers correspond to Algorithm 1 in Zeiler 2012
      # ADADELTA: AN ADAPTIVE LEARNING RATE METHOD
      # http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
          
      # Line 4: accumulate gradient
      squared.grad <<- rho * squared.grad + (1 - rho) * g^2
      
      # Line 5: compute update. RMS(x) is calculated here as 
      #   `sqrt(x + epsilon)` to prevent zero values in the denominator.
      delta <<- -sqrt(squared.delta + epsilon) / sqrt(squared.grad + epsilon) * g
      
      # Line 6: accumulate updates
      squared.delta <<- rho * squared.delta + (1 - rho) * delta^2
    }
  )
)
