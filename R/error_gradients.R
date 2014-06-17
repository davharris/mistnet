binomialLossGrad = function(y, yhat, n){
  (n * yhat - y) / (yhat - yhat^2)
}

