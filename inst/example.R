devtools::load_all()
load("birds.Rdata")
library(fastICA)

n.layer1 = 20L
n.layer2 = 4L

f = fastICA(scale(env), 10)
xx = f$S[in.train, ]

net = network$new(
  x = xx,
  y = route.presence.absence[in.train, ],
  layers = list(
    createLayer(
      dim = c(ncol(env), n.layer1),
      learning.rate = 1E-4,
      momentum = 0,
      prior = gaussian.prior(mean = 0, var = .0001),
      dataset.size = nrow(env),
      nonlinearity.name = "rectify",
      dropout = FALSE
    ),
    createLayer(
      dim = c(n.layer1, n.layer2),
      learning.rate = 1E-4,
      momentum = 0,
      prior = gaussian.prior(mean = 0, var = .0001),
      dataset.size = nrow(env),
      nonlinearity.name = "linear"
    ),
    createLayer(
      dim = c(n.layer2, ncol(route.presence.absence)),
      learning.rate = 1E-4,
      momentum = 0,
      prior = gaussian.prior(mean = 0, var = .0001),
      dataset.size = nrow(env),
      nonlinearity.name = "sigmoid"
    )
  ),
  loss = crossEntropy,
  lossGradient = crossEntropyGrad,
  minibatch.size = 50L,
  n.layers = 3L,
  n.importance.samples = 1L
)

net$layers[[3]]$biases = qlogis(colMeans(route.presence.absence))

net$layers[[1]]$coefficients[,] = rt(length(net$layers[[1]]$coefficients), df = 3) / 10
net$layers[[2]]$coefficients[,] = rt(length(net$layers[[2]]$coefficients), df = 3) / 10
net$layers[[3]]$coefficients[,] = rt(length(net$layers[[3]]$coefficients), df = 3) / 10

net$fit(1)

losses = numeric(1E4)
for(i in 1:(length(losses))){
  net$fit(1)
  losses[i] = mean(rowSums(net$reportLoss()))
  if(i%%100 == 0){
    for(layer in 1:net$n.layers){
      net$layers[[layer]]$momentum = min((1 + i / 10000) / 2, .99)
      net$layers[[layer]]$learning.rate = 2E-4 / (1 + 1E-5 * i) * (1 - net$layers[[layer]]$momentum)
    }
    cat(i, "\n")
    net$layers[[3]]$prior$mean = 
      c(0 * net$layers[[3]]$coefficients + rowMeans(net$layers[[3]]$coefficients))
    if(i %% 1000 == 0){
      plot(losses[1:1E4], type = "l")
      abline(h = mean(losses[(i-100):i]))
    }
  }
}
plot(losses, type = "l")

zz = apply(net$layers[[3]]$coefficients, 2, function(x) x / sqrt(sum(x^2)))
z = crossprod(zz)
dimnames(z) = list(colnames(route.presence.absence), colnames(route.presence.absence))
head(sort(z[,"Yellow-headed Blackbird"], decreasing=TRUE), 11)[-1]
head(sort(z[,"Veery"], decreasing=TRUE), 11)[-1]

net$minibatch.size = nrow(xx)
net$minibatch.ids = 1:nrow(xx)
net$feedForward()

mean(rowSums(net$reportLoss()))
plot(prcomp(net$layers[[2]]$output))



net$feedForward(f$S)
library(ggplot2)
color = predict(prcomp(net$layers[[2]]$output))[,1]
qplot(
  latlon[,1],
  latlon[,2],
  color = color,
  cex = 2
) + coord_equal() + scale_color_gradient2()
