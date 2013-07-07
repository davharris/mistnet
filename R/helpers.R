`%plus%` = function(matrix, vector){
  t(t(matrix) + vector)
}

rectify = function(x) pmax(x, 0)
