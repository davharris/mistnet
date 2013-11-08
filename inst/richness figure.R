library(poibin)

set.seed(1)
load("birds.Rdata")
#load("expected.values.Rdata")
expected.values = t(apply(prediction.array, 1, rowMeans))
max.richness = ncol(route.presence.absence)

richness = rowSums(route.presence.absence[in.train, ][!in.fold, ])
expected.richness = rowSums(expected.values)

# Gaussian approximation. Switch in qpoibin later
richness.sd = sqrt(
  rowSums(expected.values * (1 - expected.values))
)

lower.ci = expected.richness - qnorm(.975) * richness.sd
upper.ci = expected.richness + qnorm(.975) * richness.sd
inside.ci = ((richness < upper.ci) & (richness > lower.ci))

plot(
  expected.richness, 
  richness, 
  cex = .6, 
  asp = 1, 
  xlim = range(c(lower.ci, upper.ci)), 
  ylab = "observed.richness"#,
  #pch = ifelse(inside.ci, 1, 19)
)
abline(0,1)
abline(mean(richness.sd) * 1.96, 1, lty = 2, lwd = 2, col = "darkgray")
abline(mean(richness.sd) * -1.96, 1, lty = 2, lwd = 2, col = "darkgray")

# This is also a Guassian approximation, but it's probably not very good.
z = sqrt(mean(richness.sd^2) + mean(apply(prediction.array, 1, function(x) var(colSums(x)))))

abline(z * 1.96, 1, lty = 3, lwd = 2)
abline(z * -1.96, 1, lty = 3, lwd = 2)



p.values = sapply(
  1:nrow(expected.values), 
  function(i){
    ppoibin(richness[i], expected.values[i, ])
  }
)

n.bars = 100
hist(
  p.values * 100, 
  breaks = seq(0, 1, length = n.bars + 1) * 100, 
  xlab = "Percentile of predicted distribution", 
  col = "gray", 
  border = "#888888", 
  yaxs = "i", 
  freq = FALSE, 
  main = ""
)
abline(h = 1 / 100, lty = 2)