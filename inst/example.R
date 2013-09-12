devtools::load_all()
load("birds.Rdata")
library(fastICA)

n.ranef = 25L
n.importance.samples = 25L

n.layer1 = 50L
n.layer2 = 20L

minibatch.size = 50L

f = fastICA(scale(env), 10)
xx = f$S[in.train, ]

net = mistnet(
  x = xx,
  y = route.presence.absence[in.train, ],
  nonlinearity.names = c("rectify", "rectify", "sigmoid"),
  hidden.dims = c(n.layer1, n.layer2),
  priors = list(
    gaussian.prior(mean = 0, var = .001),
    gaussian.prior(mean = 0, var = .001),
    gaussian.prior(mean = 0, var = .01)
  ),
  learning.rate = 1E-3,
  momentum = .5,
  loss = crossEntropy,
  lossGrad = crossEntropyGrad,
  minibatch.size = minibatch.size,
  n.importance.samples = n.importance.samples,
  n.ranef = n.ranef,
  ranefSample = gaussianRanefSample,
  training.iterations = 0L
)

net$layers[[3]]$biases = qlogis(colMeans(route.presence.absence))

net$layers[[1]]$coefficients[,] = rt(length(net$layers[[1]]$coefficients), df = 3) / 10
net$layers[[2]]$coefficients[,] = rt(length(net$layers[[2]]$coefficients), df = 3) / 10
net$layers[[3]]$coefficients[,] = rt(length(net$layers[[3]]$coefficients), df = 3) / 10

net$fit(1)

losses = numeric(1E4)
for(i in 1:(length(losses))){
  net$fit(1)
  #losses[i] = mean(rowSums(net$reportLoss()))
  if(i%%100 == 0){
    for(layer.num in 1:net$n.layers){
      net$momentum = min((1 + i / 1000) / 2, .99)
      net$learning.rate = 2E-3 / (1 + 1E-5 * i) * (1 - net$momentum)
      
      # Hack to keep rectified units alive
      if(identical(net$layers[[layer.num]]$nonlinearity,rectify)){
        for(hidden.num in 1:net$layers[[layer.num]]$coef.dim[[2]]){
          if(max(net$layers[[layer.num]]$activations[,hidden.num,]) < 0){
            net$layers[[layer.num]]$biases[[hidden.num]] = 
              net$layers[[layer.num]]$biases[[hidden.num]] + .1
          }
        }
      }
    }
    cat(i, "\n")
    net$layers[[3]]$prior$mean = 
      c(0 * net$layers[[3]]$coefficients + rowMeans(net$layers[[3]]$coefficients))
  }
}
#plot(losses, type = "l")

scaled.coefs = sapply(
  1:net$layers[[3]]$coef.dim[[1]],
  function(i){
    net$layers[[3]]$coefficients[i, ] * sd(net$layers[[2]]$outputs[,i,])
  }
)
zz = apply(scaled.coefs, 1, function(x) x / sqrt(sum(x^2)))
z = crossprod(zz)
dimnames(z) = list(colnames(route.presence.absence), colnames(route.presence.absence))
head(sort(z[,"Yellow-headed Blackbird"], decreasing=TRUE), 11)[-1]
head(sort(z[,"Veery"], decreasing=TRUE), 11)[-1]

#mean(rowSums(net$reportLoss()))
plot(prcomp(net$layers[[2]]$outputs[,,1]))


net$selectMinibatch(1:nrow(f$S))
for(i in 1:n.importance.samples){
  net$feedForward(
    cbind(
      f$S, 
      net$ranefSample(nrow = nrow(f$S), ncol = n.ranef)
    ),
    i
  )
}
library(ggplot2)
color = predict(prcomp(net$layers[[2]]$outputs[,,1]))[,1]
qplot(
  latlon[,1],
  latlon[,2],
  color = color,
  cex = 2
) + coord_equal() + scale_color_gradient2()
