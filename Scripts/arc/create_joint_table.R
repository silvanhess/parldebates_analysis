# libraries --------------------------------------------------------------

library(tidyverse)

# join business info -----------------------------------------------------

transcripts_cleaned <- readRDS("Data/transcripts_cleaned.rds")
businesses_cleaned <- readRDS("Data/businesses_cleaned.rds")
subjects <- readRDS("Data/subjects.rds")

df <- businesses_cleaned |>
  distinct(BusinessShortNumber, Title, BusinessTypeName) |>
  left_join(subjects) |>
  select(
    IdSubject,
    BusinessShortNumber,
    Title,
    BusinessTypeName,
    PublishedNotes
  )

transcripts_long <- left_join(transcripts_cleaned, df, by = join_by(IdSubject))

# number_of_businesses_per_transcript <- transcripts_long |>
#   count(ID) |>
#   arrange(desc(n))

# pivot_wider
transcripts_wide <- transcripts_long |>
  group_by(ID) |>
  mutate(business_number = row_number()) |>
  ungroup() |>
  pivot_wider(
    names_from = business_number,
    values_from = c(
      BusinessShortNumber,
      Title,
      BusinessTypeName,
      PublishedNotes
    ),
    names_sep = "_"
  )

# transcripts_wide |>
#   count(PublishedNotes_1) |>
#   arrange(desc(n))

# transcripts_wide |>
#   count(BusinessTypeName_1) |>
#   arrange(desc(n))

saveRDS(transcripts_wide, "Data/transcripts_wide.rds")
