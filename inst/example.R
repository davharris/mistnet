env = as.data.frame(x[ , grep("^bio", colnames(x))])

# Number of seconds to fit the model during CV before stopping to evaluate fit
cv.seconds = 200

# How many samples to generate when evaluating CV fit
n.prediction.samples = 1000L

# Number of times to do fit & evaluate loop. Total training time is thus up to
# cv.seconds * n.iterations
n.iterations = 3L


output.df = data.frame(
  fold.id = integer(),
  seconds = numeric(),
  loglik = numeric()
)


network$methods(
  update_all = function(n.steps){
    # Update optimization hyperparameters
    .self$momentum = min((1 + .self$completed.iterations / 1000) / 2, .99)
    .self$learning.rate = 2 * starting.rate / 
      (1 + 1E-5 * .self$completed.iterations) * (1 - .self$momentum)
    
    # Hack to keep rectified units alive: add a small amount to biases of "dead"
    # units whose activations are always negative
    for(layer.num in 1:.self$n.layers){
      if(identical(.self$layers[[layer.num]]$nonlinearity,rectify)){
        for(hidden.num in 1:.self$layers[[layer.num]]$coef.dim[[2]]){
          if(max(.self$layers[[layer.num]]$activations[,hidden.num,]) < 0){
            .self$layers[[layer.num]]$biases[[hidden.num]] = 
              .self$layers[[layer.num]]$biases[[hidden.num]] + .1
          }
        }
      }
    }
    
    .self$fit(n.steps)
  }
)

setup = function(){
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
  
  # Fill things in so fitting doesn't break on the very first iteration
  net$selectMinibatch()
  net$estimateGradient()
  
  net
}

evaluate = function(){
  prediction.array = predict(
    net,
    scale(env)[in.train, ][!in.fold, ], 
    n.prediction.samples
  )
  
  cv.losses = apply(
    prediction.array,
    3,
    function(x){
      rowSums(
        net$loss(
          y = route.presence.absence[in.train, ][!in.fold, ], 
          yhat = x
        )
      )
    }
  )
  
  mean(findLogExpectedLik(cv.losses))
}


i = 0
for(fold.id in 1:max(fold.ids)){
  in.fold = fold.ids != fold.id
  net = setup()
  
  for(iteration in 1:n.iterations){
    start.time = Sys.time()
    while(
      as.double(Sys.time() - start.time, units = "secs") < cv.seconds
    ){
      net$update_all(10L)
    }
    
    i = i + 1
    output.df[i, ] = c(fold.id, iteration * cv.seconds, evaluate())
  }
}
