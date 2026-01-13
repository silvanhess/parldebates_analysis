# libraries --------------------------------------------------------------

library(tidyverse)

# import data ------------------------------------------------------------

documents_with_topics <- read_csv("Data/documents_with_topics.csv") |>
  mutate(
    transcript_id = as.character(transcript_id),
    topic = as.character(topic)
  )
topic_info <- read_csv("Data/topic_info.csv") |>
  mutate(topic = as.character(Topic)) |>
  select(-Topic)
paragraphs_classified_wide <- read_rds("Data/paragraphs_classified_wide.rds")

# plot political parties per topic ---------------------------------------

documents_with_topics_wide <- documents_with_topics |>
  left_join(topic_info, by = join_by(topic))

transcripts_topics_wide <- paragraphs_classified_wide |>
  inner_join(
    documents_with_topics_wide,
    by = join_by(transcript_id),
    relationship = "many-to-one"
  )

#
