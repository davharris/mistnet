# What about the variance of the random effects?

devtools::load_all()
load("birds.Rdata")
load("fold.ids.Rdata")
source("inst/BBS evaluation/setup.R")

cv.seconds = 100
n.prediction.samples = 500L
start.time = Sys.time()
i = 3

env = x[,1:8]

in.fold = fold.ids != i
net = mistnet(
  x = scale(env)[in.train, ][in.fold, ],
  y = route.presence.absence[in.train, ][in.fold, ],
  nonlinearity.names = c("rectify", "linear", "sigmoid"),
  hidden.dims = c(n.layer1, n.layer2),
  priors = list(
    gaussian.prior(mean = 0, var = prior.var1),
    gaussian.prior(mean = 0, var = prior.var2),
    gaussian.prior(mean = 0, var = prior.var3)
  ),
  learning.rate = starting.rate,
  momentum = .5,
  loss = crossEntropy,
  lossGrad = crossEntropyGrad,
  minibatch.size = minibatch.size,
  n.importance.samples = n.importance.samples,
  n.ranef = n.ranef,
  ranefSample = gaussianRanefSample,
  training.iterations = 0L
)

# Initialize coefficients and biases
net$layers[[3]]$biases = qlogis(
  colMeans(route.presence.absence[in.train, ][in.fold, ])
)
net$layers[[1]]$coefficients[,] = rt(prod(net$layers[[1]]$coef.dim), df = 5) / 5
net$layers[[2]]$coefficients[,] = rt(prod(net$layers[[2]]$coef.dim), df = 5) / 5
net$layers[[3]]$coefficients[,] = rt(prod(net$layers[[3]]$coef.dim), df = 5) / 5

# Fill things in so while loop stuff below doesn't break
net$selectMinibatch()
net$estimateGradient()

while(
  as.double(Sys.time() - start.time, units = "secs") < cv.seconds
){
  # Update optimization hyperparameters
  net$momentum = min((1 + net$completed.iterations / 1000) / 2, .99)
  net$learning.rate = 2 * starting.rate / 
    (1 + 1E-5 * net$completed.iterations) * (1 - net$momentum)
  
  # Hack to keep rectified units alive
  for(layer.num in 1:net$n.layers){
    if(identical(net$layers[[layer.num]]$nonlinearity,rectify)){
      for(hidden.num in 1:net$layers[[layer.num]]$coef.dim[[2]]){
        if(max(net$layers[[layer.num]]$activations[,hidden.num,]) < 0){
          net$layers[[layer.num]]$biases[[hidden.num]] = 
            net$layers[[layer.num]]$biases[[hidden.num]] + .1
        }
      }
    }
  }
  
  net$fit(10)
  cat(net$completed.iterations, "\n")
}

cat(dim(net$layers[[1]]$inputs))
prediction.array = predict(
  net,
  scale(env)[in.train, ][!in.fold, ], 
  n.prediction.samples
)
dim(net$layers[[1]]$inputs)


pp = apply(prediction.array, 2, rowMeans)

cv.losses = apply(
  prediction.array,
  3,
  function(x){
    rowSums(net$loss(y = route.presence.absence[in.train, ][!in.fold, ], yhat = x))
  }
)

cv.llik = mean(findLogExpectedLik(cv.losses))

