# libraries --------------------------------------------------------------

library(tidyverse)
library(httr)
library(jsonlite)
library(dotenv)

# create dataset for labeling --------------------------------------------

transcripts_cleaned <- readRDS("Data/transcripts_cleaned.rds")

# create a balanced dataset for labeling
# for the initial training of the classifier we want to have
# 1000 paragraphs labeled with an even distribution between french and german
# also: we want to have both climate related and not climate related paragraphs
# to achieve this, we will sample more paragraphs from businesses that have a
# higher chance of having climate related paragraphs
# final dataset should have:

# 50/50 distribution between french and german
# 70/30 distribution between Climate Businesses and not Climate Businesses

groups_before <- transcripts_cleaned |>
  count(ClimateBusiness, LanguageOfText) |>
  mutate(percentage = n / sum(n) * 100)

set.seed(1234)
transcripts_sampled <- transcripts_cleaned |>
  mutate(
    cb_weight = if_else(ClimateBusiness == TRUE, 10, 1),
    lang_weight = if_else(LanguageOfText == "FR", 3, 1),
    weight = cb_weight * lang_weight
  ) |>
  slice_sample(
    n = 1000,
    weight_by = weight
  )

groups_after <- transcripts_sampled |>
  count(ClimateBusiness, LanguageOfText) |>
  mutate(percentage = n / sum(n) * 100)

saveRDS(transcripts_sampled, "Data/transcripts_sampled.rds")

# Translage french parapraphs to german for handcoding ---------------------

transcripts_sampled <- readRDS("Data/transcripts_sampled.rds")

# # count characters in french
# transcripts_sampled |>
#   filter(LanguageOfText == "FR") |>
#   summarise(total_characters = sum(Textlength))

# Load Deepl Credentials
load_dot_env()

deepl_translate <- function(text, auth_key = Sys.getenv("DEEPL_API_KEY")) {
  url <- "https://api.deepl.com/v2/translate"

  response <- POST(
    url,
    body = list(
      auth_key = auth_key,
      text = text,
      target_lang = "EN"
    ),
    encode = "form"
  )

  # error handling
  if (response$status_code != 200) {
    stop(
      "DeepL API Fehler: ",
      status_code(response),
      " - ",
      content(response, "text")
    )
  }

  # parse
  result <- content(response, as = "parsed")

  # extract text
  result$translations[[1]]$text
}

transcripts_sampled$paragraph_translated <- map_chr(
  transcripts_sampled$paragraph,
  deepl_translate,
  .progress = TRUE
)

handcoding_dataset <- transcripts_sampled |>
  select(ID, ClimateBusiness, LanguageOfText, paragraph_translated)

saveRDS(handcoding_dataset, "Data/handcoding_dataset.rds")

# export to csv for labeling in label-studio
write_csv(handcoding_dataset, "Data/handcoding_dataset.csv")
