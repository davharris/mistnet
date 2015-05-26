`%notin%` = function(a, b){!(a %in% b)}

library(geosphere) # for regularCoordinates
library(raster)    # for worldclim
library(ape)       # for read.tree
library(geiger)    # for fitContinuous


Year = 2014

# environment -------------------------------------------------------------

retained_layers = c(2, 3, 5, 8, 9, 15, 16, 18)
wc = getData("worldclim", var = "bio", res = 10)[[retained_layers]]


# BBS ---------------------------------------------------------------------
if (!file.exists("db.sqlite")) {
  library(ecoretriever)
  
  # Version 1.7 of retriever has a problem with the current BBS web site.
  download.file(
    "https://raw.githubusercontent.com/weecology/retriever/master/scripts/bbs50stop.py",
    "~/.retriever/scripts/bbs50stop.py",
    method = "libcurl"
  )
  install("BBS50", "sqlite", data_dir = "bbs_data", db_file = "db.sqlite")
}


library(dplyr)
library(magrittr)
library("RSQLite")


# Extract valid data components

db = src_sqlite("db.sqlite")
routes  = tbl(db, "BBS50_routes") %>% filter_(~ RouteTypeID == 1 & routetypedetailid == 1)

species = tbl(db, "BBS50_species") %>% collect
  
weather = tbl(db, "BBS50_weather") %>% filter_(~runtype == 1 & rpid == 101 & year == Year)

valid_routedataids = weather %>% select(routedataid) %>% collect %>% extract2(1)

counts  = tbl(db, "BBS50_counts") %>% filter_(~RouteDataID %in% valid_routedataids)

# Get the RouteDataID for runs associated with each transect run
runs = inner_join(
  counts %>% 
    select_("RouteDataID", "countrynum", "statenum", "Route", "year") %>% 
    distinct_() %>% 
    rename_(route = "Route"),
  routes,
  c("countrynum", "statenum", "route")
) %>% collect()

# train-test split --------------------------------------------------------

# All the points within inner.radius of a center point will be in the test set.
# Everything more than outer.radius away will be in the training set.
# radii are in meters
centers = regularCoordinates(16)
inner.radius = 1.0E5
outer.radius = 2.5E5

dists = pointDistance(centers, cbind(runs$loni, runs$lati), longlat = TRUE)

runs$in_train = apply(
  dists, 
  2,
  function(x) min(x) > outer.radius
)

in_test = apply(
  dists, 
  2,
  function(x) min(x) < inner.radius
)

valid_runs = runs[runs$in_train | in_test, ]

# Identify occupants ------------------------------------------------------

rdids = valid_runs %>% extract2("RouteDataID")

collected_counts = counts %>% 
  collect %>% 
  filter_(~RouteDataID %in% rdids) %>% 
  as.data.frame

occurrences_by_rdid = lapply(
  rdids,
  function(x){
    collected_counts[x == collected_counts$RouteDataID, "AOU"]
  }
)
names(occurrences_by_rdid) = rdids

aous = occurrences_by_rdid %>% 
  unlist %>% 
  unique %>% 
  sort

# find valid species ------------------------------------------------------

# drop French column: bad unicode causes problems
french = grep("french", colnames(species))
included_species = species[match(aous, species$AOU), -french]

# presence-absence matrix -------------------------------------------------

# list of presence/absence vectors by site
pa_list = lapply(
  occurrences_by_rdid, 
  function(x){
    included_species$AOU %in% x
  }
)

pa = structure(
  do.call(rbind, pa_list),
  dimnames = list(
    names(occurrences_by_rdid), 
    included_species$spanish_common_name
  )
)


# Collapse subspecies -----------------------------------------------------

# Subspecies have three names separated by two spaces.
# For our purposes, "Colaptes auratus auratus x auratus cafer" is a subspecies
# because both parents of the hybrid belonged to the same species
subspecies = c(
  grep("^\\S* \\S* \\S*$", colnames(pa), value = TRUE),
  "Colaptes auratus auratus x auratus cafer"
)

full_species = gsub("^(\\S* \\S*).*", "\\1", subspecies)

# If a subspecies is present, the full species is present too
for(i in 1:length(subspecies)){
  pa[, full_species[i]] = pmax(pa[, subspecies[i]], pa[, full_species[i]])
}

# Drop subspecies
pa = pa[ , colnames(pa) %notin% subspecies]


# Impute hybrids and "slash" species as 50% of each parent ------------------------------------

# If a hybrid can live there, then both parent species could have been there.
# If an observer narrows it down to two species, treat it as 50-50

hybrids = grep(" [x/] ", colnames(pa), value = TRUE)
for(i in 1:length(hybrids)){
  parent1 = gsub(" x .*", "", hybrids[i])
  parent2 = gsub("\\S* x ", "", hybrids[i])
  
  # If a parent was absent, it is now considered 50% present
  # The "pmax" means that presence of a hybrid never *decreases* one's belief that the 
  # parent species was present.
  pa[ , parent1] = pmax(pa[ , parent1], pa[ , hybrids[i]] / 2)
  pa[ , parent2] = pmax(pa[ , parent2], pa[ , hybrids[i]] / 2)
}

pa = pa[ , colnames(pa) %notin% hybrids]



# Identify possible meanings of “sp." -------------------------------------

# only include species in the presence/absence matrix
included_species = included_species[included_species$spanish_common_name %in% colnames(pa), ]


# I'm interpreting "Crow sp." as "Corvus sp."
colnames(pa)[colnames(pa) == "\\\"Crow\\\" sp."] = "Corvus sp."


sps = grep("^[A-Z].* sp\\.", colnames(pa), value = TRUE)

change_list = lapply(
  sps,
  function(x){
    regex = paste0(strsplit(x, " ")[[1]][[1]], "[^/]*[^\\.]$")
    colnames(pa[ , grep(regex, colnames(pa))])
  }
)
names(change_list) = sps


# I'm using common names for the unidentified Tern because the family-level taxonomy has changed since
# the data was recorded.
change_list$`\\\"Tern\\\" sp.` = 
  included_species[grep("[^\\.] Tern$", included_species$english_common_name), ]$spanish_common_name

# Also using common names for ravens
change_list$`\\\"Raven\\\" sp.` = 
  included_species[grep("[^\\.] Raven$", included_species$english_common_name), ]$spanish_common_name

# I'm limiting the term "gull" to the genera "Larus", "Chroicocephalus", and "Leucophaeus"
change_list$`\\\"Gull\\\" sp.` = 
  included_species[included_species$genus %in% c("Larus", "Chroicocephalus", "Leucophaeus"), ]$spanish_common_name

# "woodpecker" is any member of the family
change_list$`\\\"Woodpecker\\\" sp.` = 
  included_species[included_species$family == "Picidae", ]$spanish_common_name

# Ardeids could be any member of the family
change_list$`\\\"Ardeid\\\" sp.` = 
  included_species[included_species$family == "Ardeidae", ]$spanish_common_name

# Ditto for trochilids
change_list$`\\\"Trochilid\\\" sp.` = 
  included_species[included_species$family == "Trochilidae", ]$spanish_common_name


# Do we have all the "quoted" species?
stopifnot(all(grep("\\\"", colnames(pa), value = TRUE) %in% names(change_list)))

# "quoted" species don't belong inside the lists, just as names of the list elements
change_list = lapply(
  change_list, function(x){
    grep("^[A-Z]", x, value = TRUE)
  }
)


# Impute partial observations ---------------------------------------------

run_dists = pointDistance(
  cbind(valid_runs$loni, valid_runs$lati), 
  cbind(valid_runs$loni, valid_runs$lati),
  longlat = TRUE, 
  allpairs = TRUE
)

sigma = 1000 * 1000 # 1000 kilo-meters

k = exp(-0.5 * run_dists^2 / sigma^2)

for(i in 1:length(change_list)){
  name = names(change_list)[i]
  sp_names = change_list[[i]]
  
  intensities = sapply(
    change_list[[i]], 
    function(x){colSums(k * pa[,x]) / colSums(k)}
  )
  p = intensities / rowSums(intensities)
  
  imputed = pmax(p, pa[, sp_names])
  
  uncertain_rows = as.logical(pa[ , name])
  
  pa[uncertain_rows , sp_names] = imputed[uncertain_rows, sp_names]
}

# Drop columns from change_list
pa = pa[ , colnames(pa) %notin% names(change_list)]


# Import phylogeny -------------------------------------------------------------

tre = read.tree("AllBirdsEricson1.tre", n = 1)

phylo_names = gsub("_", " ", tre$tip.label)

# # merge synonyms ----------------------------------------------------------
# 
# library(taxize)
# 
# matchless = colnames(pa)[!(colnames(pa) %in% gsub("_", " ", tre$tip.label))]
# 
# syns = synonyms(matchless, db = c("itis"), accepted = FALSE)
# 
# na_matches = names(which(is.na(syns)))
# 
# for(bbs_name in names(syns)){
#   if(length(syns[[bbs_name]]) == 2){
#     syns[[bbs_name]]$bbs_name = bbs_name
#   }
# }
# 
# syns_df = bind_rows(syns[sapply(syns, function(x) length(x) > 1)])
# 
# syns_df$name = gsub("^(\\S* \\S*).*", "\\1", syns_df$name)
# 
# renames = syns_df[syns_df$name %in% phylo_names, ]
# 
# fixed_by_renames = sapply(
#   syns[!is.na(syns)],
#   function(x){
#     gsub("^(\\S* \\S*).*", "\\1", x$name) %in% phylo_names
#   }
# )
# 
# 
# failed_matches = sapply(
#   syns[!is.na(syns)], 
#   function(x){any(x$name %in% phylo_names)}
# ) %>% 
#   not %>% 
#   which %>% 
#   names
# 
# 


# Wrap everything up ------------------------------------------------------

phylogeny = drop.tip(tre, which(!(gsub("_", " ", tre$tip.label) %in% colnames(pa))))
pa = pa[ , match(gsub("_", " ", phylogeny$tip.label), colnames(pa))]
species = included_species[match(colnames(pa), included_species$spanish_common_name),]

colnames(pa) = gsub(" ", "_", colnames(pa))

runs = runs[match(row.names(pa), runs$RouteDataID), ]

env = raster::extract(
  raster::getData("worldclim", var = "bio", res = "10"), 
  as.matrix(runs[, c("loni", "lati")])
)


save(
  phylogeny, 
  pa, 
  env, 
  runs, 
  species, 
  file = "phylo-birds.Rdata"
)
