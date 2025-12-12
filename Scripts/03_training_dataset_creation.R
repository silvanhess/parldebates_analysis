# libraries --------------------------------------------------------------

library(tidyverse)
# library(swissparl)
library(httr)
library(jsonlite)
library(dotenv)

# create dataset for labeling --------------------------------------------

transcripts_wide <- readRDS("Data/transcripts_wide.rds")

# create a balanced dataset for labeling
# 50% climate related, 50% not climate related
# 50% german, 50% french
# so 25% per group

groups_before <- transcripts_wide |> count(ClimateBusiness, LanguageOfText)

set.seed(1234)
transcripts_sampled <- transcripts_wide |>
  group_by(ClimateBusiness, LanguageOfText) |>
  slice_sample(n = 250) |>
  ungroup()

groups_after <- transcripts_sampled |> count(ClimateBusiness, LanguageOfText)

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
