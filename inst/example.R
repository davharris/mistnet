devtools::load_all()
load("birds.Rdata")

start.time = Sys.time()

hyperparameters = list(
  n.ranef = 25L,
  n.importance.samples = 25L,
  n.layer1 = 50L,
  n.layer2 = 20L,
  minibatch.size = 50L,
  prior.var1 = 1,
  prior.var2 = .001,
  prior.var3 = .001,
  starting.rate = 1E-3,
  cv.seconds = 10
)

in.fold = TRUE

net = mistnet(
  x = scale(env)[in.train, ][in.fold, ],
  y = route.presence.absence[in.train, ][in.fold, ],
  nonlinearity.names = c("rectify", "rectify", "sigmoid"),
  hidden.dims = c(hyperparameters$n.layer1, hyperparameters$n.layer2),
  priors = list(
    gaussian.prior(mean = 0, var = hyperparameters$prior.var1),
    gaussian.prior(mean = 0, var = hyperparameters$prior.var2),
    gaussian.prior(mean = 0, var = hyperparameters$prior.var3)
  ),
  learning.rate = hyperparameters$starting.rate,
  momentum = .5,
  loss = crossEntropy,
  lossGrad = crossEntropyGrad,
  minibatch.size = hyperparameters$minibatch.size,
  n.importance.samples = hyperparameters$n.importance.samples,
  n.ranef = hyperparameters$n.ranef,
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

while(
  as.double(Sys.time() - start.time, units = "secs") < hyperparameters$cv.seconds
){
  net$fit(10)
  
  for(layer.num in 1:net$n.layers){
    net$momentum = min((1 + net$completed.iterations / 1000) / 2, .99)
    net$learning.rate = 2 * hyperparameters$starting.rate / 
      (1 + 1E-5 * net$completed.iterations) * (1 - net$momentum)
    
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
  cat(net$completed.iterations, "\n")
}
