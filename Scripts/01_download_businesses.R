# libraries --------------------------------------------------------------

library(tidyverse)
library(swissparl)
library(furrr)

# get businesses -------------------------------------------

# für alle Geschäfte, die im Parlament von 2016-2025 behandelt wurden,
# sind auch Geschäfte nötig, die bereits vor 2016 eingereicht wurden.
# Daher werden hier alle Geschäfte von 2010-2025 heruntergeladen.

yr = 2010:2025

# with walk

get_businesses <- function(yr) {
  folder <- "Data/businesses"
  if (!dir.exists(folder)) {
    dir.create(folder)
  }

  out_file <- file.path(folder, paste0(yr, ".rds"))
  if (file.exists(out_file)) {
    return(NULL)
  } # skip existing files

  dt <- get_data(
    "Business",
    SubmissionDate = c(
      paste(">", yr, "-01-01", sep = ""),
      paste("<", yr, "-12-31", sep = "")
    ),
    Language = "EN"
  )

  saveRDS(dt, file.path(folder, paste0(yr, ".rds")))
}

walk(
  yr,
  get_businesses,
  .progress = TRUE
)

businesses <- map_dfr(
  list.files("Data/businesses", full.names = T),
  readRDS
)
saveRDS(businesses, "Data/businesses.rds")
