f = function(tuple.size){
  
  spp = sample.int(368, tuple.size)
  cv.losses = apply(
    prediction.array,
    3,
    function(x){
      rowSums(
        net$loss(
          y = route.presence.absence[in.train, spp, drop = FALSE][!in.fold, ], 
          yhat = x[ , spp, drop = FALSE]
        )
      )
    }
  )
  
  my.llik = mean(findLogExpectedLik(cv.losses))
  ind.llik = -mean(
    rowSums(
      net$loss(
        y = route.presence.absence[in.train, spp, drop = FALSE][!in.fold, ], 
        yhat = pp[ , spp, drop = FALSE]
      )
    )
  )
  my.llik - ind.llik
}

indices = c(1, 5, 10, 25, 50, 100, 200, 300, ncol(route.presence.absence))
lik.ratios = exp(sapply(indices, f))

plot(
  indices, 
  lik.ratios * 1.01^indices, 
  log = "y", 
  xlab = "# of species to predict", 
  ylab = "likelihood ratio", 
  axes = FALSE, 
  xaxs = "i",
  xlim = c(0, ncol(route.presence.absence) + 1),
  type = "n"
)
axis(1, 50 * c(0:10))
axis(2, c(10^c(0:5)), labels = paste0("10^", 0:5), las = 3)
lines(indices, 1.01^indices, lty = 2, col = "darkgray", lwd = 3)
lines(indices, rep(1, length(indices)), lty = 3, col = "darkgray", lwd = 2)
lines(
  indices, 
  lik.ratios * 1.01^indices, 
  type = "o",
  cex = .5, 
  pch = 16,
  lwd = 2
)
