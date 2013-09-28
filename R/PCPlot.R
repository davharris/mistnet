PCPlot = function(x, npcs = min(10, length(x$sdev))){
  var = x$sd^2
  
  y = cumsum(var) / sum(var)
  
  plot(
    x = 0:npcs, 
    y = c(0, y[1:npcs]), 
    ylim = c(0, 1), 
    yaxs = "i", 
    xaxs = "i",
    type = "o"
  )
  abline(h = seq(0, 1, by = .1), lty = 2, col = "lightgray")
}
