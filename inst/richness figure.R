set.seed(1)
load("birds.Rdata")
load("expected.values.Rdata")

richness = rowSums(route.presence.absence[in.test, ])
expected.richness = rowSums(expected.values)
richness.sd = sqrt(
  rowSums(expected.values * (1 - expected.values))
)

lower.ci = expected.richness - qnorm(.999) * richness.sd
upper.ci = expected.richness + qnorm(.999) * richness.sd
outside.ci = ((richness < upper.ci) & (richness > lower.ci))

plot(
  expected.richness, 
  richness, 
  cex = .6, 
  asp = 1, 
  xlim = range(c(lower.ci, upper.ci)), 
  ylab = "observed.richness",
  pch = ifelse(outside.ci, 1, 19)
)
abline(0,1)
