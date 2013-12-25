load("birds.Rdata")
load("mistnet.predictions.Rdata")
devtools::load_all()
null.predictions = matrix(
  NA, 
  nrow = sum(in.test), 
  ncol = ncol(route.presence.absence),
  dimnames = list(NULL, colnames(route.presence.absence))
)

for(species in colnames(route.presence.absence)){
  null.predictions[ , species] = mean(route.presence.absence[in.train, species])
}

save(null.predictions, file = "null.predictions.Rdata")
