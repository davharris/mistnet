initializeMars = function(x, y){
  
  marsmodel = mars(x, y)
  
  cuts = marsmodel$cuts[marsmodel$selected.terms[-1], ]
  
  initial.biases = - rowSums(
    cuts * sign(marsmodel$factor[marsmodel$selected.terms[-1], ])
  )
  initial.weights = t(marsmodel$factor[marsmodel$selected.terms[-1] , ])
  
  rbind(Intercept = initial.biases, initial.weights)
}

# e
# h1 = rectify(train.x %*% w1 %plus% b1)
# stopifnot(
#   all.equal(model.matrix(marsmodel)[,-1], h1, check.attributes = FALSE)
# )