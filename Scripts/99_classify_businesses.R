# assign the business to the transcript (join business and transcript tabless)

# libraries --------------------------------------------------------------

library(tidyverse)

# join business and transcript table -------------------------------------

transcripts_cleaned <- readRDS("Data/transcripts_cleaned.rds")
businesses_cleaned <- readRDS("Data/businesses_cleaned.rds")
subjects <- readRDS("Data/subjects.rds")

transcripts_subjects <- left_join(
  transcripts_cleaned,
  subjects,
  by = join_by(IdSubject),
  relationship = "many-to-many"
)

transcripts_subjects_businesses <- left_join(
  transcripts_subjects,
  businesses_cleaned,
  by = join_by(BusinessShortNumber),
  relationship = "many-to-one"
)

unique_businesses <- transcripts_subjects_businesses |>
  distinct(BusinessShortNumber) |>
  pull(BusinessShortNumber)

businesses_debated <- businesses_cleaned |>
  filter(BusinessShortNumber %in% unique_businesses)

missing_businesses <- anti_join(
  transcripts_subjects_businesses,
  businesses_cleaned,
  by = join_by(BusinessShortNumber)
) |>
  distinct(BusinessShortNumber, Title)

# get missing businesses -------------------------------------------------


