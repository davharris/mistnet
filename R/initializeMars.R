marsInitializer = function(x, y){
  
  marsmodel = mda::mars(x, y)
  
  cuts = marsmodel$cuts[marsmodel$selected.terms[-1], ]
  
  initial.biases = - rowSums(
    cuts * sign(marsmodel$factor[marsmodel$selected.terms[-1], ])
  )
  initial.weights = t(marsmodel$factor[marsmodel$selected.terms[-1] , ])
  
  list(
    biases = initial.biases, 
    weights = initial.weights, 
    n = length(initial.biases)
  )
}
