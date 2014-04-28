devtools::load_all()
load("inst/fakedata.Rdata")

n.importance.samples = 25L
minibatch.size = 21L
n.ranef = 4L
learning.rate = .1

net = network$new(
  x = env[ , 1:3],
  y = fakedata,
  layers = list(
    createLayer(
      n.inputs = 3L + n.ranef,
      n.outputs = ncol(fakedata),
      prior = gaussian.prior$new(mean = 0, var = 1),
      minibatch.size = minibatch.size,
      n.importance.samples = n.importance.samples,
      nonlinearity.name = "sigmoid",
      updater.name = "adagrad",
      updater.arguments = list(
        learning.rate = learning.rate
      )
    )
  ),
  n.layers = 1L,
  dataset.size = nrow(env),
  minibatch.size = minibatch.size,
  n.importance.samples = n.importance.samples,
  loss = crossEntropy,
  lossGradient = crossEntropyGrad,
  ranefSample = gaussianRanefSample,
  n.ranef = n.ranef,
  completed.iterations = 0L
)

# reset/initialize layer state
net$layers[[1]]$resetState(minibatch.size, n.importance.samples)

# initialize biases
net$layers[[1]]$biases = matrix(qlogis(colMeans(fakedata)), nrow = 1)


# Fill in all the stuff for the MRF nonlinearity
scale = mean(abs(lateral[upper.tri(lateral)]))
net$layers[[1]]$nonlinearity = mf_mrf.nonlinearity(
  lateral = matrix(0, nrow = ncol(fakedata), ncol = ncol(fakedata)),
  maxit = 50L,
  damp = 0.2,
  tol = 1E-4,
  updater = new(
    "adagrad.updater",
    delta = matrix(0, nrow = ncol(fakedata), ncol = ncol(fakedata)),
    learning.rate = learning.rate / 10
  ),
  l1.decay = 1 / scale / nrow(env)
)




for(iter in 1:5){
  new.order = sample.int(nrow(net$x))
  net$x = net$x[new.order, ]
  net$y = net$y[new.order, ]
  for(i in 1:50){
    if(i%%50 == 0){
      cat("\n")
    }
    cat(".")
    net$fit(10)
    if(is.nan(net$layers[[1]]$outputs[[1]])){
      stop("NaNs detected :-(")
    }
  }
}




plot(net$layers[[1]]$coefficients[1:3, ], coefs[1:3, ], asp = 1); abline(0,1)
plot(
  net$layers[[1]]$nonlinearity$lateral,
  lateral, 
  cex = .8,
  col = "#00000020",
  pch = 16
)
abline(0,1)
abline(0,0)
abline(v = c(-.05, .05))

summary(lm(coefs[4, ] ~ 0 + t(net$layers[[1]]$coefficients[4:(3 + n.ranef), ])))
summary(lm(coefs[5, ] ~ 0 + t(net$layers[[1]]$coefficients[4:(3 + n.ranef), ])))

square = function(x){x^2}
1-mean(square(net$layers[[1]]$nonlinearity$lateral - lateral))/mean(square(lateral))

mean(abs(net$layers[[1]]$nonlinearity$lateral - lateral))

