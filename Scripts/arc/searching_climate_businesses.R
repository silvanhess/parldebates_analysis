# Description ------------------------------------------------------------

# Search Businesses related to Climate Change

# libraries --------------------------------------------------------------

library(tidyverse)
library(swissparl)
library(furrr)

# Approach 1: Using the Search function of the API -------------------------

search_strings <- c(
  "CO2",
  "Erneuerbare",
  "Heizung",
  "Klimafonds",
  "Solar",
  "Treibhaus",
  "Wärmepumpe",
  "Wasserkraft",
  "Windkraft",
  "Emissionshandel",
  "Energieeffizienz",
  "Energiespeicher",
  "Elektromobilität"
)

get_businesses <- function(search_strings) {
  folder <- "Data/businesses_climate"
  if (!dir.exists(folder)) {
    dir.create(folder)
  }

  out_file <- file.path(folder, paste0(search_strings, ".rds"))
  if (file.exists(out_file)) {
    return(NULL)
  } # skip existing files

  dt <- get_data(
    table = "Business",
    Title = paste0("~", search_strings),
    Language = "DE"
  )

  saveRDS(dt, file.path(folder, paste0(search_strings, ".rds")))
}

walk(
  search_strings,
  get_businesses,
  .progress = TRUE
)

businesses_climate_1 <- map_dfr(
  list.files("Data/businesses_climate", full.names = T),
  readRDS
)

saveRDS(businesses_climate_1, "Data/businesses_climate.rds")

# analyze business tags
businesses_long_1 <- businesses_climate_1 |>
  separate_rows(Tags, TagNames, sep = "\\|")

business_tags_1 <- businesses_long_1 |> count(Tags, TagNames)
business_tags_1 |> arrange(desc(n)) |> print(n = Inf)

# Approach 2: Filtering Businesses --------------------------------------------------

businesses_cleaned <- readRDS("Data/businesses_cleaned.rds")

businesses_cleaned[1, ]

search_strings <- c(
  "CO2",
  "Erneuerbare",
  "Heizung",
  "Klimafonds",
  "Solar",
  "Treibhaus",
  "Wärmepumpe",
  "Wasserkraft",
  "Windkraft",
  "Emissionshandel",
  "Energieeffizienz",
  "Energiespeicher",
  "Elektromobilität"
)

businesses_climate_2 <- businesses_cleaned |>
  # filter(str_detect(Title, paste(search_strings, collapse = "|")))
  filter(str_detect(
    paste(Title, BusinessDetails),
    paste(search_strings, collapse = "|")
  ))
