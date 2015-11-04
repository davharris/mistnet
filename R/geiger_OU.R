# #' @import geiger
# geiger_OU = function(phy, dat){
#   oufit =  fitContinuous(
#     phy, 
#     model = "OU", 
#     dat, 
#     SE=NA, 
#     ncores=NULL
#   )
#   
#   K = vcv(rescale(phy, "OU", alpha = oufit$opt$alpha, sigsq = oufit$opt$sigsq))
#   
#   K + diag(ncol(K)) * oufit$opt$SE
# }
