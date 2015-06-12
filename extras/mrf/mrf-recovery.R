set.seed(1)
library(beepr)

maxit = 2500
tick_size = 10

n_random_env = 5L

devtools::load_all()
library(progress)
load("extras/mrf/fakedata.Rdata")

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
          learning.rate = .1
        ),
        prior = logistic.prior(location = 0, scale = 0.05)
      ),
      size = ncol(fakedata),
      sampler = gaussian.sampler(ncol = n_random_env, sd = 1),
      prior = gaussian.prior(mean = 0, sd = 1)
    )
  ),
  loss = bernoulliLoss(),
  updater = adagrad.updater(learning.rate = .1),
  initialize.biases = TRUE, 
  initialize.weights = TRUE,
  sampler = gaussian.sampler(ncol = n_random_env, sd = 1)
)


pb <- progress_bar$new(
  format = "  Fitting [:bar] :percent eta: :eta",
  total = maxit,
  clear = FALSE
)
 
par(mfrow = c(1, 2))
pb$tick(0)
for(i in 1:(maxit / tick_size)){
  net$fit(tick_size)
  
  net$layers[[1]]$prior$update(
    net$layers[[1]]$weights,
    update.mean = FALSE,
    update.sd = TRUE,
    min.sd = .1
  )
  lateral_estimates = net$layers[[1]]$nonlinearity$lateral
  upper_lateral_estimates = lateral_estimates[upper.tri(lateral_estimates)]
  
  
  if(any(is.na(net$layers[[1]]$nonlinearity$lateral))){
    stop()
  }
  
  
  if(i %% 10 == 0){
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
      col = "#00000008",
      pch = 16,
      asp = 1
    )
    abline(0,1)
    abline(0,0)
    abline(v = 0)
  }
  
  pb$tick(tick_size)
}
beep()



summary(lm(coefs[1, ] ~ net$layers[[1]]$weights[1, ]))
summary(lm(coefs[2, ] ~ net$layers[[1]]$weights[2, ]))
summary(lm(coefs[3, ] ~ net$layers[[1]]$weights[3, ]))

pcs = predict(
  prcomp(
    t(net$layers[[1]]$weights[4:(3 + n_random_env), ])
  )
)
summary(lm(coefs[4, ] ~ ., data = as.data.frame(pcs)))
summary(lm(coefs[5, ] ~ ., data = as.data.frame(pcs)))

cor(lateral[upper.tri(lateral)], upper_lateral_estimates, method = "spearman")

library(quantreg)
summary(
  rq(lateral[upper.tri(lateral)] ~ upper_lateral_estimates), 
  se = "ker"
)


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
  col = "#00000010"
)

library(rags2ridges)
ridge = ridgeS(cov(fakedata), 5/nrow(fakedata))
plot(
  ridge[upper.tri(ridge)],
  lateral[upper.tri(lateral)]
)

plot(
  cor(fakedata)[upper.tri(cor(fakedata))],
  lateral[upper.tri(lateral)]
)
