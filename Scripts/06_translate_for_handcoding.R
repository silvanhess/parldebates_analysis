# libraries --------------------------------------------------------------

library(tidyverse)
library(httr)
library(jsonlite)
library(dotenv)

# Translage  for handcoding ----------------------------------------------

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
  select(ID, ClimateBusiness, LanguageOfText, paragraph, paragraph_translated)

# handcoding_dataset <- readRDS("Data/handcoding_dataset.rds")
# transcripts_sampled <- readRDS("Data/transcripts_sampled.rds")
# handcoding_dataset <- inner_join(handcoding_dataset, transcripts_sampled, by = join_by(ID)) |> 
#   select(ID, ClimateBusiness.x, LanguageOfText.x, paragraph, paragraph_translated) |> 
#     rename(ClimateBusiness = ClimateBusiness.x, LanguageOfText = LanguageOfText.x)

# export to csv for labeling in label-studio
write_csv(handcoding_dataset, "Data/handcoding_dataset.csv")