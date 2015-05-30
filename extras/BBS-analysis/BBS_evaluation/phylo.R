warning("fix the ordering of pa so it matches the ordering of tip.labels")

library(ape)
library(geiger)
devtools::load_all()
load("phylo-birds.Rdata")
library(fastICA)

set.seed(1)

ic_env = fastICA(scale(env), 6)$S

net = mistnet(
  x = ic_env[runs$in_train, ],
  y = pa[runs$in_train, ],
  layer.definitions = list(
    defineLayer(
      nonlinearity = rectify.nonlinearity(),
      size = 50,
      prior = gaussian.prior(mean = 0, sd = 0.1)
    ),
    defineLayer(
      nonlinearity = rectify.nonlinearity(),
      size = 10,
      prior = gaussian.prior(mean = 0, sd = 0.1)
    ),
    defineLayer(
      nonlinearity = sigmoid.nonlinearity(),
      size = ncol(pa),
      prior = gaussian.prior(mean = 0, sd = 0.1)
    )
  ),
  loss = bernoulliRegLoss(1 + 1E-6, 1 + 1E-6),
  updater = adagrad.updater(learning.rate = 0.1),
  initialize.weights = FALSE,
  initialize.biases = FALSE
)

# Initialize weights and biases
glorot = function(n_j, n_j_plus_1){
  sqrt(6) / (n_j + n_j_plus_1)
}
glorot_range = sapply(net$layers, function(layer){sqrt(6)/sum(sqrt(dim(layer$weights)))})
for(i in 1:length(net$layers)){
  layer = net$layers[[i]]
  layer$weights[] = runif(length(layer$weights), -glorot_range[[i]], glorot_range[[i]])
}


# Positive biases for relu units -> less sparse
net$layers[[1]]$biases[] = 1
net$layers[[2]]$biases[] = 1

# weakly regularized initialization for layer 3 biases
net$layers[[3]]$biases[] = qlogis((colSums(pa) + 2) / (nrow(pa) + 4))


completed_epoch = 0 
max_epochs = 10
message("fitting neural network...")
while(net$row.selector$completed.epochs < max_epochs){
  net$fit(iterations = 10)
  cat(".")
  
  # Revive "dead" hidden units whose means are always zero
  for(i in 1:ncol(layer$weights)){
    if(mean(layer$outputs[,i,]) == 0){
      layer$biases[i] = layer$biases[i] + 0.1
    }
  }
  
  # Update 3rd layer's means
  net$layers[[3]]$prior$update(weights = net$layers[[3]]$weights, 
                               update.mean = TRUE, 
                               update.sd = FALSE, 
                               min.sd = NULL
  )
  
  if(net$row.selector$completed.epochs != completed_epoch){
#     
#     message("updating phylogenetic priors...")
#     K_array = array(NA, c(ncol(pa), ncol(pa), 10))
#     noise_sd = rep(0, dim(K_array)[3])
#     for(i in 1:dim(K_array)[3]){
#       fit = fitContinuous(
#         phy = phylogeny, 
#         dat = structure(net$layers[[3]]$weights[i, ], names = colnames(pa)), 
#         SE = NA, 
#         model = "OU"
#       )
#       K_array[,,i] = vcv(rescale(phylogeny, "OU", alpha = fit$opt$alpha, sigsq = fit$opt$sigsq))
#       noise_sd[i] = fit$opt$SE
#     }
#     net$layers[[3]]$prior = gp.prior(
#       K = K_array, 
#       noise_sd = pmax(noise_sd, .01), 
#       coefs = net$layers[[3]]$weights
#     )
#     message("updating other priors...")
    
    cat("Completed epoch", completed_epoch, "\n")
    # Update prior variance
    for(layer in net$layers[1:3]){
      layer$prior$update(
        layer$weights, 
        update.mean = FALSE, 
        update.sd = TRUE,
        min.sd = .01
      )
    }
    message("fitting neural network...")
  }
  completed_epoch = net$row.selector$completed.epochs
}
net$completed.iterations

z = predict(net, ic_env[!runs$in_train, ], 100)
zz = apply(z, 2, rowMeans)
colnames(z) = colnames(zz) = colnames(pa)

i = 9
hist(
  cov2cor(
    net$layers[[3]]$prior$K[,,i] + net$layers[[3]]$prior$noise_sd[i]^2 * diag(ncol(pa))
  )^2,
  ylim = c(0, 1000)
)

round(sapply(1:10,function(i){
  quantile(cov2cor(
    net$layers[[3]]$prior$K[,,i] + net$layers[[3]]$prior$noise_sd[i]^2 * diag(ncol(pa))
  )^2, .995)
}) * 100)


x = scale(net$layers[[3]]$weights[i, ])[match(colnames(pa), phylogeny$tip.label)]
names(x) = phylogeny$tip.label

phylogeny_ = phylogeny
phylogeny_$tip.label = rep("-", ncol(pa)) 
plot(
  phylogeny_,
  tip.color = rgb(plogis(x), .5, plogis(-x)),
  no.margin = TRUE
)

plot(net$layers[[3]]$weights[i, ], col = factor(species$sporder))


data = data.frame(
  Genus_species = gsub(" ", "_", names(x)), 
  Reg = 1L + (species[match(names(x), species$spanish_common_name), "sporder"] == "Passeriformes"), 
  X = x
)
phylo_ = phylogeny_
phylo_$tip.label = as.character(data$Genus_species)
o = OUwie(phylo_, data = data, model = "OU1")


str(tree)
