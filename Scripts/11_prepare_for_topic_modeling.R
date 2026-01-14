# libraries --------------------------------------------------------------

library(tidyverse)

# import data ------------------------------------------------------------

# paragraphs_classified_wide <- read_rds("Data/paragraphs_classified_wide.rds")
paragraphs_climate <- read_rds("Data/paragraphs_climate.rds")
paragraphs_words_cleaned <- read_rds("Data/paragraphs_words_cleaned.rds")

# include all transcripts that belong to a climate relevant business------

paragraphs_climate_summary <- paragraphs_climate |>
  distinct(paragraph_id, paragraph) |> 
  mutate(paragraph_length = str_count(paragraph, "\\S+"))
transcripts_climate_summary <- paragraphs_climate |>
  group_by(transcript_id) |>
  summarise(
    transcript_text = paste(paragraph, collapse = " "),
    transcript_word_length = str_count(transcript_text, "\\S+"),
    .groups = "drop"
  ) |> 
  filter(transcript_word_length >= 50) # filter very short transcripts

# # without stopwords
# transcripts_climate <- paragraphs_climate |>
#   # filter(climate_business == TRUE) |>
#   distinct(transcript_id, paragraph_id, paragraph) |>
#   # left_join(
#   #   paragraphs_words_cleaned,
#   #   by = join_by(paragraph_id),
#   #   relationship = "many-to-many"
#   # ) |>
#   # select(-paragraph) |>
#   group_by(transcript_id) |>
#   summarise(
#     transcript_text = paste(word, collapse = " "),
#     transcript_word_length = str_count(transcript_text, "\\S+"),
#     .groups = "drop"
#   )

# # print random transcript text
# transcripts_climate |>
#   slice_sample(n = 10) |>
#   pull(transcript_text)

write_csv(
  transcripts_climate_summary,
  "Data/transcripts_climate_for_topic_modeling.csv"
)
