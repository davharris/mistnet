`%notin%` = function(a, b){!(a %in% b)}

library(ecoretriever) # for install
library(geosphere)    # for regularCoordinates
library(raster)       # for worldclim
library(NMF)          # for non-negative matrix factorization
library(dplyr)        # for data manipulation
library(magrittr)     # for data manipulation
library("RSQLite")    # for data extraction
library(parallel)     # for mclapply
library(traits)       # for birdlife data
library(beepr)        # for audible progress notification

Year = 2014

iucn_id = function(sciname , silent = FALSE){
  spec <- tolower(sciname)
  spec <- gsub(" ", "-", spec)
  url <- paste("http://api.iucnredlist.org/go/", spec, sep = "")
  e <- try(readLines(url), silent = silent)
  
  id_plus = grep("http://www.iucnredlist.org/apps/redlist/details/", e, value = TRUE)
  gsub(".*/", "", id_plus)
  
}



# environment -------------------------------------------------------------

retained_layers = c(2, 3, 5, 8, 9, 15, 16, 18)
wc = getData("worldclim", var = "bio", res = 10)[[retained_layers]]


# BBS ---------------------------------------------------------------------
if (!file.exists("db.sqlite")) {
  
  # Version 1.7 of retriever has a problem with the current BBS web site.
  download.file(
    "https://raw.githubusercontent.com/weecology/retriever/master/scripts/bbs50stop.py",
    "~/.retriever/scripts/bbs50stop.py",
    method = "libcurl"
  )
  install("BBS50", "sqlite", data_dir = "bbs_data", db_file = "db.sqlite")
}


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


# Import traits -------------------------------------------------------------

if(!file.exists("iucns.csv")){
  iucns = mclapply(
    colnames(pa), 
    function(x){
      id = iucn_id(x)
      c(x, ifelse(length(id) == 1, id, NA))
    }
  )
  iucn_matrix = do.call(rbind, iucns)
  iucns = data_frame(name = iucn_matrix[,1], id = iucn_matrix[,2])
  beepr::beep()
  write.csv(iucns, file = "iucns.csv", row.names = FALSE)
}else{
  iucns = read.csv("iucns.csv", stringsAsFactors = FALSE)
}

iucns = na.omit(iucns)

if(!file.exists("habitats.csv")){
  habitats = lapply(
    iucns[[2]],
    function(x){
      if(!is.na(x)){
        message(x)
        birdlife_habitat(x)
      }
    }
  )
  beepr::beep()
  habitats = bind_rows(habitats)
  write.csv(habitats, "habitats.csv", row.names = FALSE)
}else{
  habitats = read.csv("habitats.csv", stringsAsFactors = FALSE)
}

habitats = merge.data.frame(habitats, iucns, by = "id")

habitats = habitats[habitats$Importance %in% c("major", "suitable"), ]
habitats = habitats[habitats$Occurrence %in% c("breeding", "resident"), ]
habitats$bilevel = paste(habitats$Habitat..level.1., habitats$Habitat..level.2., sep = ": ")

habitats$Importance = 1 + (habitats$Importance == "major")

traits = iucns[ , FALSE]
for(level_name in c("Habitat..level.1.", "Habitat..level.2.", "bilevel")){
  habs = unique(habitats[[level_name]])
  for(hab in habs[nchar(habs) > 0]){
    hab_names = list(
      unique(habitats[habitats[[level_name]] == hab & habitats$Importance == 1, "name"]),
      unique(habitats[habitats[[level_name]] == hab & habitats$Importance == 2, "name"])
    )
    
    traits[iucns$name %in% hab_names[[1]], hab] = 1
    traits[iucns$name %in% hab_names[[2]], hab] = 2
    traits[is.na(traits[[hab]]), hab] = 0
  }
}

max_cor = 0.85

traits = traits[ , colSums(traits > 0) > 10]

# Function to drop variables that are collinear with target variable
dropcor = function(x, target, max_cor){
  traits[ , cor(traits)[, target] < max_cor | colnames(traits) == target]  
}

ordered_categories = lapply(
  sort(unique(colSums(traits)), decreasing = TRUE),
  function(x){
    tied_columns = colnames(traits)[colSums(traits) == x]
    tied_columns[order(nchar(tied_columns))]
  }
)



for(trait in unlist(ordered_categories)){
  # If the trait is still in the matrix, check it for collinear variables
  if(trait %in% colnames(traits)){
    traits = dropcor(traits, trait, max_cor = max_cor)
  }
}

stopifnot(
  sum(cor(traits)[upper.tri(cor(traits))] > max_cor) == 0
)
  

K = 15

set.seed(1)

nmf_object = nmf(traits, K, nrun = 10)

for(i in 1:K){
  message(i)
  row = coef(nmf_object)[i, ]
  print(round(sort(row[row > 1E-8], decreasing = TRUE), 2))
}

trait_ids = c(
  temperate_shrubland = 1,
  coastal = 2,
  warm_dry_shrubland = 3,
  warm_moist_montane_forest = 4,
  human = 5,
  marine = 6,
  freshwater = 7,
  temperate_forest = 8,
  wetlands = 9,
  boreal_forest = 10,
  subtropical_forest = 11,
  agricultural = 12, 
  rocky_desert = 13, 
  temperate_grassland = 14, 
  other_grassland = 15
)
stopifnot(all(trait_ids == 1:K))
trait_names = names(trait_ids)


summary(lm(unlist(traits) ~ c(basis(nmf_object) %*% coef(nmf_object))))

prior_means = round(apply(basis(nmf_object), 2, function(traits) traits / sd(traits)), 8)
rownames(prior_means) = iucns$name
colnames(prior_means) = trait_names

nmf_coefficients = round(coef(nmf_object), 8)

row.names(nmf_coefficients) = trait_names

pa = pa[ , traits$name]

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

species = included_species[match(colnames(pa), included_species$spanish_common_name),]

runs = runs[match(row.names(pa), runs$RouteDataID), ]

env = raster::extract(
  raster::getData("worldclim", var = "bio", res = "10"), 
  as.matrix(runs[, c("loni", "lati")])
)


save(
  prior_means, 
  nmf_coefficients,
  nmf_object,
  pa, 
  env, 
  runs, 
  species, 
  file = "birds-traits.Rdata"
)

