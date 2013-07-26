devtools::load_all()
library(ggplot2)
load("../scrf/birds.Rdata")

y = route.presence.absence[in.train, ]
x = scale(env)[in.train, ]
n.out = ncol(y)
n.hid = 50
n.bottleneck = 10
n.in = ncol(x)
n = nrow(y)
mini.n = 100

lr = .05

w1 = matrix(0, nrow = ncol(x), ncol = n.hid)
w1[,] = rt(length(w1), 2)/10

w2 = matrix(0, nrow = n.hid, ncol = n.bottleneck)
w2[,] = rt(length(w2), 2)/10

w3 = matrix(0, nrow = n.bottleneck, ncol = n.out)
w3[,] = runif(length(w3), -.01, .1)


b1 = rep(0, n.hid)
b3 = qlogis(colMeans(y))

maxit = 10000

errors = rep(NA, maxit/100)

#Rprof()

for(i in 1:maxit){
  
  in.batch = sample.int(n, mini.n)
  batch.x = x[in.batch, ]
  batch.y = y[in.batch, ]
  
  h = sigmoid(batch.x %*% w1 %plus% b1)
  bottleneck.h = h %*% w2
  yhat = sigmoid(bottleneck.h %*% w3 %plus% b3)
  
  
  delta3 = crossEntropyGrad(y = batch.y, yhat = yhat) * 
    sigmoidGrad(s = yhat)
  
  dw3 = matrixMultiplyGrad(
    n.hid = n.bottleneck, 
    n.out = n.out, 
    delta = delta3,
    h = bottleneck.h
  )
    
  delta2 = delta3 %*% t(w3)
  dw2 = matrixMultiplyGrad(
    n.hid = n.hid, 
    n.out = n.bottleneck, 
    delta = delta2,
    h = h
  )
  
  delta1 = sigmoidGrad(s = h) * (delta2 %*% t(w2))
  dw1 = t(
    matrixMultiplyGrad(
      n.hid = n.in, 
      n.out = n.hid, 
      delta = delta1,
      h = batch.x
    )
  )
  
  b1 = b1 - colMeans(delta1) * lr
  b3 = b3 - colMeans(delta3) * lr
  
  dw1 = dw1/mini.n - w1 * 1E-4
  w1 = w1 + dw1 * lr
  
  dw2 = t(dw2)/mini.n - w2 * 1E-3
  w2 = w2 + dw2 * lr
  
  dw3 = t(dw3)/mini.n - w3 * 1E-3
  w3 = w3 + dw3 * lr
  
  if(i%%100 == 0){
    error = crossEntropy(y = batch.y, yhat = yhat)
    errors[i/100] = mean(rowSums(error))
    message(i)
  }
  if(is.na(dw1[[1]])){stop("NAs")}
}

# Rprof(NULL)
# summaryRprof()

plot(1:length(errors) * 100, errors, type = "l")



h = sigmoid(scale(env) %*% w1 %plus% b1)
bottleneck.h = h %*% w2
yhat = sigmoid(bottleneck.h %*% w3 %plus% b3)


color = predict(prcomp(bottleneck.h))[,1]
qplot(
  latlon[,1], 
  latlon[,2], 
  color = color, 
  cex = 2
) + 
  coord_equal() + 
  scale_color_gradient2(limits = range(predict(prcomp(bottleneck.h))))



hist(w2 %*% w3)

