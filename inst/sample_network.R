load("../scrf/birds.Rdata")

y = route.presence.absence[in.train, ]
x = scale(env)[in.train, ]
n.out = ncol(y)
n.hid = 5
n.in = ncol(x)
n = nrow(y)
mini.n = 100

lr = .1

w1 = matrix(0, nrow = ncol(x), ncol = n.hid)
w1[,] = runif(length(w1), -.001, .001)

w2 = matrix(0, nrow = n.hid, ncol = n.out)
w2[,] = runif(length(w2), -.001, .001)

b1 = rep(0, n.hid)
b2 = qlogis(colMeans(y))




maxit = 100000

errors = rep(NA, maxit)
dw1 = 0
dw2 = 0

for(i in 1:maxit){
  
  in.batch = sample.int(n, mini.n)
  batch.x = x[in.batch, ]
  batch.y = y[in.batch, ]
  
  h = sigmoid(batch.x %*% w1 %plus% b1)
  yhat = sigmoid(h %*% w2 %plus% b2)
  error = crossEntropy(y = batch.y, yhat = yhat)
  
  delta2 = crossEntropyGrad(y = batch.y, yhat = yhat) * 
    sigmoidGrad(s = yhat)
  
  dw2 = matrixMultiplyGrad(
    n.hid = n.hid, 
    n.out = n.out, 
    delta = delta2,
    h = h
  )
  b2 = b2 - colMeans(delta2) * lr
  
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
  
  dw1 = dw1/mini.n * lr - w1 * .01
  w1 = w1 + dw1
  
  dw2 = dw2/mini.n * lr
  w2 = w2 + t(dw2) - w2 * .00001
  
  
  errors[i] = mean(error)
  if(is.na(errors[i])){stop("NAs")}
}

plot(errors, type = "l")



h = sigmoid(scale(env) %*% w1 %plus% b1)
yhat = sigmoid(h %*% w2 %plus% b2)

z = apply(h, 2, round)

color = factor(
  apply(z, 1, function(x) paste0(x, collapse = ""))
)
qplot(
  latlon[,1], 
  latlon[,2], 
  color = color, 
  cex = 2
) + coord_equal() + 
  scale_color_brewer(type = "qual")

qplot(
  latlon[,1], 
  latlon[,2], 
  color = h[,5], 
  cex = 2
) + coord_equal()


-2 * mean(
  rowSums(
    dbinom(
      route.presence.absence[in.test, ], 
      prob = yhat[in.test, ],
      size = 1, 
      log = TRUE
    )
  )
)


colnames(w2) = colnames(route.presence.absence)

scaled.w = apply(w2, 2, function(x) x / sqrt(sum(x^2)))

x = scaled.w[1, ]
y = scaled.w[3, ]
plot(x, y, type = "n")
text(x, y, label = colnames(route.presence.absence))
