set.seed(1)

devtools::load_all()
load("inst/fakedata.Rdata")

scale = mean(abs(lateral[upper.tri(lateral)]))

net = mistnet(
  x = env[ , 1:3],
  y = fakedata,
  layer.definitions = list(
    defineLayer(
      nonlinearity = mf_mrf.nonlinearity(
        lateral = matrix(0, nrow = ncol(fakedata), ncol = ncol(fakedata)),
        maxit = 50L,
        damp = 0.2,
        tol = 1E-4,
        updater = new(
          "adagrad.updater",
          delta = matrix(0, nrow = ncol(fakedata), ncol = ncol(fakedata)),
          learning.rate = .05
        ),
        l1.decay = 1 / scale / nrow(env)
      ),
      size = ncol(fakedata),
      prior = gaussian.prior(mean = 0, sd = .5)
    )
  ),
  loss = bernoulliLoss(),
  updater = adagrad.updater(learning.rate = .1),
  initialize.biases = TRUE, 
  initialize.weights = TRUE
)




par(mfrow = c(1, 2))
for(i in 1:25){
  net$fit(100)
  if(any(is.na(net$layers[[1]]$nonlinearity$lateral))){
    stop()
  }
  
  plot(
    net$layers[[1]]$weights[1:3, ], 
    coefs[1:3, ], 
    asp = 1,
    main = net$completed.iterations
  )
  abline(0,1)
  plot(
    net$layers[[1]]$nonlinearity$lateral,
    lateral, 
    cex = .8,
    col = "#00000020",
    pch = 16,
    asp = 1
  )
  abline(0,1)
  abline(0,0)
  abline(v = 0)
}

pcs = predict(prcomp(t(net$layers[[1]]$weights[4:(3 + 10), ])))

summary(lm(coefs[1, ] ~ 0 + net$layers[[1]]$weights[1, ]))
summary(lm(coefs[2, ] ~ 0 + net$layers[[1]]$weights[2, ]))
summary(lm(coefs[3, ] ~ 0 + net$layers[[1]]$weights[3, ]))

summary(lm(coefs[4, ] ~ PC1+PC2, data = as.data.frame(pcs)))
summary(lm(coefs[5, ] ~ PC1+PC2, data = as.data.frame(pcs)))

summary(lm(c(lateral) ~ 0+c(net$layers[[1]]$nonlinearity$lateral)))

mean(abs(net$layers[[1]]$nonlinearity$lateral))


par(mfrow = c(1, 1))
plot(
  net$layers[[1]]$nonlinearity$lateral,
  lateral, 
  asp = 1,
  type = "n"
)
abline(0,1, lwd  = 2, col = "darkgray")
abline(0, 0, lwd = 2, col = "darkgray")
abline(v = 0, lwd = 2, col = "darkgray")
points(
  net$layers[[1]]$nonlinearity$lateral,
  lateral,
  cex = .5,
  pch = 16,
  col = "#00000040"
)

