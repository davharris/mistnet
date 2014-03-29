source("inst/BBS evaluation/auxiliary-mistnet-methods.R")

env = as.data.frame(x[ , grep("^bio", colnames(x))])

# Number of seconds to fit the model during CV before stopping to evaluate fit
cv.seconds = 1000

# How many samples to generate when evaluating CV fit
n.prediction.samples = 500L

# Number of times to do fit & evaluate loop. Total training time is thus up to
# cv.seconds * n.iterations
n.iterations = 1L


output.df = data.frame(
  fold.id = integer(),
  seconds = numeric(),
  loglik = numeric()
)

cat("Fold number: ")

i = 0
for(fold.id in 1:max(fold.ids)){
  in.fold = fold.ids != fold.id
  net = buildNet(
    x = scale(env)[in.train, ][in.fold, ],
    y = route.presence.absence[in.train, ][in.fold, ]
  )
  
  for(iteration in 1:n.iterations){
    start.time = Sys.time()
    while(
      as.double(Sys.time() - start.time, units = "secs") < cv.seconds
    ){
      if(is.nan(net$layers[[3]]$outputs[[1]])){
        stop("NaNs detected :-(")
      }
      net$update_all(10L)
    }
    
    i = i + 1
    output.df[i, ] = c(fold.id, iteration * cv.seconds, cv.evaluate())
  }
  cat(fold.id)
}

rm(net)
