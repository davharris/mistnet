# This function determines if a species is "valid" by looking at its latin and
# common names.  It is called as part of the data-extraction script in this 
# folder

validateSpecies = function(){
  # The first 7 lines are filler.
  skip = readLines("proprietary.data/BBS/SpeciesList.txt", n = 7)
  
  # The sixth line has column names separated by lots of spaces
  column.names = grep(".$", strsplit(skip[6], " ")[[1]], value = TRUE)
  
  # The seventh line has dashes that can be used for determining column widths
  dashes = skip[7]
  
  
  # The +1 is to compensate for the spaces between columns of dashes.
  species.df = na.omit(
    read.fwf(
      "proprietary.data/BBS/SpeciesList.txt",
      skip = 7, 
      widths = nchar(strsplit(dashes, " ")[[1]]) + 1,
      as.is = TRUE,
      colClasses = "character",
      strip.white = TRUE
    )
  )
  colnames(species.df) = column.names
  rownames(species.df) = species.df$AOU
  
  # the bottom of the file says there should be 1465 rows.
  stopifnot(nrow(species.df) == 1465)
  
  
  # species with slashes, " or ", " x " or " X " or " sp." "unid", etc.
  # are either unknown or hybrids.
  pattern = "/|( or )|( X )|( x )|( sp\\.)|(unid)|hybrid|Admin Code"
  
  bad.latin = grep(
    pattern,
    species.df$Spanish_Common_Name
  )
  
  bad.english = grep(
    pattern,
    species.df$English_Common_Name
  )
  
  # Something is a potential subspecies if it has three words separated by
  # spaces.
  possible.subspecies = grep("^.* .* .*$", species.df$Spanish_Common_Name)
  subspecies.ID = possible.subspecies[
    is.na(match(possible.subspecies, bad.latin))
  ]
  subspecies = species.df$Spanish_Common_Name[subspecies.ID]
  
  # binomial nomenclature version of the subspecies name (i.e., discard the
  # third name)
  subspecies.binomial = sapply(
    strsplit(subspecies, " "), 
    function(x){
      paste(x[1:2], collapse = " ")
    }
  )
  
  
  
  # I've also decided to throw out species that have associated subspecies
  # (e.g., with half a dozen junco races, the unknown-race-junco isn't really
  # informative either)
  has.subspecies = which(species.df$Spanish_Common_Name %in% subspecies.binomial)
  
  bad.species = sort(
    unique(c(bad.latin, bad.english, subspecies.ID, has.subspecies))
  )
  
  # get rid of leading 0's in AOU, since they're not used elsewhere
  rownames(species.df) = gsub("^0", "", rownames(species.df))
  
  return(species.df[-bad.species, ])
}

