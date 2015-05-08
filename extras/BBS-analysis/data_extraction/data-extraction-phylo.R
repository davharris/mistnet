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
  
  # Version 1.7 of retriever has a 
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

# Identify occupants ------------------------------------------------------
collected_counts = counts %>% collect %>% as.data.frame

rdids = unique(collected_counts$RouteDataID)

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
french = grep("french", colnames(species)) # bad unicode causes problems
included_species = species[match(aous, species$AOU), -french]

# species with slashes, " or ", " x " or " X " or " sp." "unid", etc.
# are either unknown or hybrids.
bad_species = "/| or | X | x | sp\\.|unid|hybrid|Admin Code"

# Bad genera include a bad species and start with a capital letter
bad_genera = included_species %>% 
  extract2("spanish_common_name") %>%
  grep(bad_species, ., value = TRUE) %>%
  strsplit(" ") %>% 
  unlist %>% 
  grep("^[A-Z]", ., value = TRUE) %>%
  unique %>%
  paste0(collapse = "|")

row_valid = !grepl(bad_species, included_species$spanish_common_name) & 
  !grepl(bad_genera, included_species$spanish_common_name)

valid_species = included_species %>% filter_(~row_valid)


# Find rows with subspecies (three Latin names)
subspecies_rows = grep(" .+ ", valid_species$spanish_common_name)

# Find the corresponding rows for the whole species
species_rows = valid_species[subspecies_rows, ] %>% 
  extract(c("genus", "species")) %>%
  apply(1, paste, collapse = " ") %>% 
  match(valid_species$spanish_common_name)

# Replace subspecies rows with species rows
valid_species[subspecies_rows, ] = valid_species[species_rows, ]

valid_aous = unique(valid_species$AOU)

# presence-absence matrix -------------------------------------------------

# list of presence/absence vectors by site
pa_list = lapply(
  occurrences_by_rdid, 
  function(x){
    valid_aous %in% x
  }
)

pa = structure(
  do.call(rbind, pa_list),
  dimnames = list(
    names(occurrences_by_rdid), 
    valid_species$spanish_common_name[match(valid_aous, valid_species$AOU)]
  )
)



# Phylogenies -------------------------------------------------------------

tre = read.tree("AllBirdsEricson1.tre", n = 1)

# x = structure(
#   rnorm(length(tre$tip.label)),
#   names = tre$tip.label
# )
# 
# 
# fit = fitContinuous(tre, dat = x, SE = NA, model = "white")


# train-test split --------------------------------------------------------

# All the points within inner.radius of a center point will be in the test set.
# Everything more than outer.radius away will be in the training set.
# radii are in meters
centers = regularCoordinates(12)
inner.radius = 1.5E5
outer.radius = 3E5

dists = pointDistance(centers, latlon, longlat = TRUE)

in.train = apply(
  dists, 
  2,
  function(x) min(x) > outer.radius
)

in.test = apply(
  dists, 
  2,
  function(x) min(x) < inner.radius
)
