# libraries --------------------------------------------------------------

library(tidyverse)

# import data ------------------------------------------------------------

labeled_dataset <- read.csv("Data/labeled_dataset.csv") |>
  select(ID, paragraph_translated, sentiment) |>
  rename(class = sentiment)

handcoding_dataset <- read.csv("Data/handcoding_dataset.csv")

training_dataset <- inner_join(
  labeled_dataset,
  handcoding_dataset,
  by = join_by(ID)
) |>
  select(paragraph, paragraph_translated.x, class) |>
  rename(paragraph_translated = paragraph_translated.x)

saveRDS(training_dataset, "Data/training_dataset.rds")

# transcripts_cleaned <- readRDS("Data/transcripts_cleaned.rds")
# transcripts_classified <- left_join(transcripts_cleaned, training_dataset, by = join_by(paragraph))
