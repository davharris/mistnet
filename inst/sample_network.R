devtools::load_all()
library(ggplot2)
library(fastICA)
load("../scrf/birds.Rdata")

ica = fastICA(poly(scale(env), degree = 2), n.comp = 20)

projected.env = ica$S

y = route.presence.absence[in.train, ]
x = projected.env[in.train, ]
n.out = ncol(y)
n.hid = 50
n.bottleneck = 10
n.in = ncol(x)
n = nrow(y)
mini.n = 2

momentum = 0.5
n.importance.samples = 20
random.effect.sd = 0.2
lr = .001

w1 = matrix(0, nrow = ncol(x), ncol = n.hid)
w1[,] = rnorm(length(w1), sd = .2) * (sample.int(2, length(w1), replace = TRUE) - 1L)

w2 = matrix(0, nrow = n.hid, ncol = n.bottleneck)
w2[,] = rnorm(length(w2), sd = .2) * (sample.int(2, length(w2), replace = TRUE) - 1L)

w3 = matrix(0, nrow = n.bottleneck, ncol = n.out)
w3[,] = rnorm(length(w3), sd = .2) * (sample.int(2, length(w3), replace = TRUE) - 1L)


w1grads = array(dim = c(nrow(w1),ncol(w1),n.importance.samples))
w2grads = array(dim = c(nrow(w2),ncol(w2),n.importance.samples))
w3grads = array(dim = c(nrow(w3),ncol(w3),n.importance.samples))

b1 = rep(0, n.hid)
b3 = qlogis(colMeans(y))


dw1 = dw2 = dw3 = 0
w1grads = array(dim = c(nrow(w1),ncol(w1),n.importance.samples))
w2grads = array(dim = c(nrow(w2),ncol(w2),n.importance.samples))
w3grads = array(dim = c(nrow(w3),ncol(w3),n.importance.samples))

maxit = 20000

errors = rep(NA, maxit/100)
importance.errors = matrix(NA, nrow = mini.n, ncol = n.importance.samples)

#Rprof()

for(i in 1:maxit){
  
  
  in.batch = sample.int(n, mini.n)
  batch.x = x[in.batch, ]
  batch.y = y[in.batch, ]
  
  w1grad = w2grad = w3grad = 0
  
  for(j in 1:n.importance.samples){
    
    h = sigmoid(batch.x %*% w1 %plus% b1)
    bottleneck.h = h %*% w2 %plus% rnorm(n.bottleneck, sd = random.effect.sd)
    yhat = sigmoid(bottleneck.h %*% w3 %plus% b3)
    
    importance.errors[ , j] = rowSums(crossEntropy(y = batch.y, yhat = yhat))
    
    delta3 = crossEntropyGrad(y = batch.y, yhat = yhat) * sigmoidGrad(s = yhat)
    delta2 = delta3 %*% t(w3)
    delta1 = sigmoidGrad(s = h) * (delta2 %*% t(w2))
    
    
    w3grads[ , , j] = t(
      matrixMultiplyGrad(
        n.hid = n.bottleneck, 
        n.out = n.out, 
        delta = delta3,
        h = bottleneck.h
      )
    )
    w2grads[ , , j] = t(
      matrixMultiplyGrad(
        n.hid = n.hid, 
        n.out = n.bottleneck, 
        delta = delta2,
        h = h
      )
    )
    w1grads[ , , j] = t(
      matrixMultiplyGrad(
        n.hid = n.in, 
        n.out = n.hid, 
        delta = delta1,
        h = batch.x
      )
    )
  }
  
  unweighted.importance.weights = exp(colSums(importance.errors) - min(colSums(importance.errors)))
  importance.weights = unweighted.importance.weights / sum(unweighted.importance.weights)
  
  for(j in 1:n.importance.samples){
    w1grad = w1grad + w1grads[ , , j] * importance.weights[j]
    w2grad = w2grad + w2grads[ , , j] * importance.weights[j]
    w3grad = w3grad + w3grads[ , , j] * importance.weights[j]
  }
  
  dw1 = w1grad / mini.n - w1 * 1E-3 + dw1 * momentum
  dw2 = w2grad / mini.n - w2 * 1E-1 + dw2 * momentum
  dw3 = w3grad / mini.n - w3 * 1E-2 + dw3 * momentum
  
  b1 = b1 - colMeans(delta1) * lr
  b3 = b3 - colMeans(delta3) * lr
  
  w1 = w1 + dw1 * lr
  w2 = w2 + dw2 * lr
  w3 = w3 + dw3 * lr
  
  
  if(i%%100 == 0){
    errors[i/100] = sum(colMeans(importance.errors) * importance.weights)
    message(i)
    momentum = pmin(1 - 10/(10 + i/100), .8)
  }
  if(is.na(dw1[[1]])){stop("NAs")}
}

# Rprof(NULL)
# summaryRprof()

niter = 1000
yhats = array(dim = c(nrow(route.presence.absence[in.test, ]), ncol(route.presence.absence), niter))
dimnames(yhats) = list(NULL, colnames(route.presence.absence), NULL)

h = sigmoid(projected.env[in.test, ] %*% w1 %plus% b1)

for(i in 1:niter){
  bottleneck.h = h %*% w2 %plus% rnorm(n.bottleneck, sd = random.effect.sd)
  yhats[,,i] = sigmoid(bottleneck.h %*% w3 %plus% b3)
}

yhat = apply(yhats, 2, rowMeans)

h = sigmoid(projected.env[in.test, ] %*% w1 %plus% b1)
bottleneck.h = h %*% w2
yhat.mle = sigmoid(bottleneck.h %*% w3 %plus% b3)
colnames(yhat.mle) = colnames(route.presence.absence)


mean(colSums(crossEntropy(route.presence.absence[in.test, ], yhat.mle)))
mean(colSums(crossEntropy(route.presence.absence[in.test, ], yhat)))

spp = c("Chipping Sparrow", "Pine Siskin")
sitenum = 3
apply(t(yhats[sitenum,spp,]), 2, sd)
plot(
  t(yhats[sitenum,spp,]), 
  xlim = c(0, 1), 
  ylim = c(0,1), 
  xaxs = "i", 
  yaxs = "i", 
  cex = .5
)
abline(0,1)
points(matrix(yhat[sitenum, spp], ncol = 2), col = 4)
points(matrix(yhat.mle[sitenum, spp], ncol = 2), col = 2)

plot(1:length(errors) * 100, errors, type = "l")