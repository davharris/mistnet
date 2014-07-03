load("birds.Rdata")
n.folds = 5L

# Setting a RNG seed so that I can ensure that all methods see the same CV 
# splits.  Then randomize the seed.
set.seed(1)
fold.ids = sample(rep(1:n.folds, length = sum(in.train)))

# Ensure that no folds have completely missing or completely present species
# during CV training.  
range.colmeans = range(
  sapply(
    1:n.folds, 
    function(i) colMeans(route.presence.absence[in.train, ][fold.ids != i, ])
  )
)
stopifnot(min(range.colmeans) > 0, max(range.colmeans) < 1)


save(fold.ids, file = "fold.ids.Rdata")
