# libraries --------------------------------------------------------------

library(tidyverse)
library(swissparl)

# create dataset for labeling --------------------------------------------

transcripts_cleaned <- readRDS("Data/transcripts_cleaned.rds")

# create a balanced dataset for labeling
# 50% climate related, 50% not climate related
# 50% german, 50% french
# so 25% per group

groups_before <- transcripts_cleaned |> count(ClimateBusiness, LanguageOfText)

set.seed(1234)
transcripts_sampled <- transcripts_cleaned |>
  group_by(ClimateBusiness, LanguageOfText) |>
  slice_sample(n = 250) |>
  ungroup()

groups_after <- transcripts_sampled |> count(ClimateBusiness, LanguageOfText)

saveRDS(transcripts_sampled, "Data/transcripts_sampled.rds")

# Translage french parapraphs to german for handcoding ---------------------

transcripts_sampled <- readRDS("Data/transcripts_sampled.rds")

# count characters in french
french_characters <- transcripts_sampled |>
  filter(LanguageOfText == "FR") |>
  summarise(total_characters = sum(Textlength))


# export to csv for labeling in doccano
transcripts_sampled |>
  select(ID, ClimateBusiness, LanguageOfText, paragraph) |>
  write_csv("Data/handcoding_dataset.csv")
