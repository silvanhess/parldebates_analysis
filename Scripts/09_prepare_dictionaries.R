# libraries --------------------------------------------------------------

library(tidyverse)
library(dotenv)
library(openxlsx)
source("Scripts/00_functions.R")

# import data -------------------------------------------------------------

keywords_extracted <- read_rds("Data/keywords_extracted.rds")

# prepare german dictionary -----------------------------------------------

german_keywords <- keywords_extracted |>
  filter(language == "DE") |>
  pull(keywords) |>
  str_split(",") |>
  unlist() |>
  str_squish() |>
  sort() |>
  unique() |>
  discard(~ .x == "")

# prepare french dictionary -----------------------------------------------

french_keywords <- keywords_extracted |>
  filter(language == "FR") |>
  pull(keywords) |>
  str_split(",") |>
  unlist() |>
  str_squish() |>
  sort() |>
  unique() |>
  discard(~ .x == "")

# translate to english ---------------------------------------------------

# prepare dataframe
german_dictionary <- tibble(
  keyword_german = german_keywords,
  keyword_english = NA
)

french_dictionary <- tibble(
  keyword_french = french_keywords,
  keyword_english = NA
)

# Load Deepl Credentials
load_dot_env()

german_dictionary$keyword_english <- map_chr(
  german_dictionary$keyword_german,
  deepl_translate,
  .progress = TRUE
)

french_dictionary$keyword_english <- map_chr(
  french_dictionary$keyword_french,
  deepl_translate,
  .progress = TRUE
)

# export ------------------------------------------------------------------

# export as excel
write.xlsx(german_dictionary, "Data/german_dictionary.xlsx")
write.xlsx(french_dictionary, "Data/french_dictionary.xlsx")

# by hand, check the keywords and clean the dictionaries
# after cleaning, save them as *_curated.xlsx
