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
abline(mean(richness.sd) * 1.96, 1, lty = 2)
abline(mean(richness.sd) * -1.96, 1, lty = 2)

richness.sd + 