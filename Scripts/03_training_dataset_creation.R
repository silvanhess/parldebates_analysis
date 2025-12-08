# libraries --------------------------------------------------------------

library(tidyverse)
library(swissparl)
library(furrr)

# find climate related businesses ----------------------------------------

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

businesses_climate_en <- businesses_climate
businesses_climate <- map_dfr(
  list.files("Data/businesses_climate", full.names = T),
  readRDS
)
saveRDS(businesses_climate, "Data/businesses_climate.rds")

# create a sub-dataset "climate" -----------------------------------------

subjects_all <- readRDS("Data/subject_businesses_2015_2025.rds")
businesses_climate <- readRDS("Data/businesses_climate.rds")

subjects_climate <- subjects_all |>
  filter(BusinessShortNumber %in% businesses_climate$BusinessShortNumber)

transcripts_cleaned <- readRDS("Data/transcripts_cleaned.rds")
transcripts_grouped <- transcripts_cleaned |>
  mutate(
    ClimateBusiness = if_else(
      IdSubject %in% subjects_climate$IdSubject,
      1,
      0
    )
  )
saveRDS(transcripts_grouped, "Data/transcripts_grouped.rds")


# join business info -----------------------------------------------------

# businesses <- readRDS("Data/businesses_2015_2025.rds") |>
#   distinct(IdSubject, BusinessShortNumber, Title, BusinessTypeName)

# left_join(
#   subjects |> select(IdSubject, BusinessShortNumber, Title),
#   by = "IdSubject"
# )
# transcripts_filtered <- transcripts_labeled |>

# create dataset for labeling --------------------------------------------

transcripts_grouped <- readRDS("Data/transcripts_grouped.rds")

# create a balanced dataset for labeling
# 50% climate related, 50% not climate related
# 50% german, 50% french
# so 25% per group

transcripts_grouped |> count(ClimateBusiness, LanguageOfText)

handcoding_dataset <- transcripts_grouped |>
  group_by(ClimateBusiness, LanguageOfText) |>
  slice_sample(n = 250) |>
  ungroup()

handcoding_dataset |> count(ClimateBusiness, LanguageOfText)

saveRDS(handcoding_dataset, "Data/handcoding_dataset.rds")

# export to csv for labeling in doccano
handcoding_dataset |>
  select(ID, ClimateBusiness, LanguageOfText, paragraph) |>
  write_csv("Data/handcoding_dataset.csv")
