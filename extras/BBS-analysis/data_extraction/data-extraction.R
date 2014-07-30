# Vector of stops to include
stops = paste0("Stop", 1:50)

# This file builds the training and test datasets for the analysis in the 
# paper.  It relies on having files from the Breeding Bird Survey
# in a top-level folder called proprietary data.

library(geosphere) # for regularCoordinates
library(raster)    # for worldclim
library(caret)     # for findCorrelations
library(lubridate) # for dates and times

year = 2011     # year of BBS data
min.train = 10  # species must be observed this many times in the training set to be included


# All the points within inner.radius of a center point will be in the test set.
# Everything more than outer.radius away will be in the training set.
# radii are in meters
centers = regularCoordinates(12)
inner.radius = 1.5E5
outer.radius = 3E5



source("extras/BBS-analysis/data_extraction/species-handling.R")

# eliminate unacceptable (e.g. non-species) taxa
valid.species.df = validateSpecies()



# Breeding Bird Survey data -----------------------------------------------

# As stated in runtype.txt, runs are only valid if RunType == 1
{
  runs = read.csv("proprietary.data/BBS/weather.csv")
  runs = runs[runs$RunType == 1, ]
}


stop.data.filenames  = dir(
  "proprietary.data/BBS/50-StopData/1997ToPresent_SurveyWide/", 
  pattern = "\\.csv$",
  full.names = TRUE
)

stop.data = do.call(
  rbind,
  lapply(
    stop.data.filenames,
    function(path){
      df = read.csv(path)
      # Only Run protocol (RPID) type 101 is standard. See RunProtocolID.txt
      # only keep data from one year
      df[df$year == year & df$RPID == 101, ]
    }
  )
)
mode(stop.data$AOU) = "character"

# eliminate invalid species.
valid.AOU = rownames(valid.species.df)
stop.data = stop.data[stop.data$AOU %in% valid.AOU, ]

# eliminate runs deemed unacceptable above
stop.data = stop.data[stop.data$RouteDataID %in% runs$RouteDataId, ] 


routeDataIDs = unique(stop.data$RouteDataID)

# Import routes, then remove bad ones:
#   only RouteTypeID 1 is roadside
#   only RouteTypeDetailID 1 and 2 are random
routes = read.csv("proprietary.data/BBS/routes.csv", header = TRUE)
routes = with(
  routes, 
  routes[RouteTypeID == 1 & RouteTypeDetailId %in% c(1,2), ]
)



# The routes file doesn't have unique identifiers, so we have to make some
makeRouteID = function(mat) apply(
  mat[,c("countrynum", "statenum", "Route")], 
  1,
  function(x) paste(x, collapse = "-")
)


# Do I really need two `match` statements?
routes = routes[
  match(
    makeRouteID(runs[match(routeDataIDs, runs$RouteDataId), ]),
    makeRouteID(routes)
  ),
]

which.runs = match(routeDataIDs, runs$RouteDataId)

ydays = yday(
  ymd(
    paste(
      year,
      runs$Month[which.runs], 
      runs$Day[match(routeDataIDs, runs$RouteDataId)], 
      sep = "-"
    )
  )
)

stopifnot(all(nchar(runs$StartTime[which.runs]) == 3))
start.times = as.numeric(
  as.difftime(
    hm(gsub("^(.)(.*)$", "\\1:\\2", runs$StartTime[which.runs]))
  )
) / 60 / 60

latlon = routes[,c("Longi", "Lati")]


# worldclim ---------------------------------------------------------------

# Will download fresh data if not already found on your machine!
worldclim.raster = getData("worldclim", var = "bio", res = 10)

env = extract(worldclim.raster, latlon)
correlations = caret::findCorrelation(cor(na.omit(env)), cutoff = .8)
env = env[ , -correlations]

# final -------------------------------------------------------------------

# produce a site by species matrix of presence-absence
route.presence.absence = sapply(
  sort(unique(stop.data$AOU)),
  function(species){
    bool = (species == stop.data$AOU) & 
      (rowSums(stop.data[, stops, drop = FALSE]) > 0)
    present.stops = stop.data$RouteDataID[bool]
    routeDataIDs %in% present.stops
  }
)

row.names(route.presence.absence) = routeDataIDs
colnames(route.presence.absence) = valid.species.df[
  sort(unique(stop.data$AOU)), 
  "English_Common_Name"
]



# handling NA environmental data
omitted = attr(na.omit(env), "na.action")
x = cbind(env, start.times, ydays)[-omitted,]
route.presence.absence = route.presence.absence[-omitted, ]
latlon = latlon[-omitted, ]



# Splitting ---------------------------------------------------------------

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


# eliminate species that occurred in fewer than min.train training stops
is.not.too.rare = colSums(route.presence.absence[in.train, ]) > min.train
route.presence.absence = route.presence.absence[ , is.not.too.rare]



# save --------------------------------------------------------------------

species.data = valid.species.df[
  match(
    colnames(route.presence.absence), 
    valid.species.df$English_Common_Name
  ), 
]
names(species.data)[names(species.data) == "Spanish_Common_Name"] = "Latin_name"
rownames(species.data) = gsub(" ", "_", species.data$English_Common_Name)

save(
  route.presence.absence = route.presence.absence,
  latlon = latlon,
  in.test = in.test,
  in.train = in.train,
  x = x,
  species.data = species.data,
  file = "birds.Rdata"
)


