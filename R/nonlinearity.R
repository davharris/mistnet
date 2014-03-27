nonlinearity = setRefClass(
  Class = "nonlinearity",
  fields = list(),
  methods = list(
    f = function(x){
      stop("activation function not defined for this nonlinearity")
    },
    grad = function(x){
      stop("gradient not defined for this nonlinearity")
    }
  )
)

linear.nonlinearity = setRefClass(
  Class = "linear.nonlinearity",
  fields = list(),
  contains = "nonlinearity",
  methods = list(
    f = function(x){x},
    grad = function(x){1}
  )
)

sigmoid.nonlinearity = setRefClass(
  Class = "sigmoid.nonlinearity",
  fields = list(),
  contains = "nonlinearity",
  methods = list(
    f = function(x){
      # Benchmarking suggests that this is 20% faster than plogis()  
      storage.mode(x) = "numeric"
      binomial()$linkinv(x)
    },
    grad = function(x){
      s = sigmoid(x)
      s * (1 - s)
    }
  )
)


rectify.nonlinearity = setRefClass(
  Class = "rectify.nonlinearity",
  fields = list(),
  contains = "nonlinearity",
  methods = list(
    f = rectify,
    grad = function(x){
      x > 0
    }
  )
)
