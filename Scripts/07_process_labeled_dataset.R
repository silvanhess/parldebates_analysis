# libraries --------------------------------------------------------------

library(tidyverse)
source("Scripts/00_functions.R")

# import data ------------------------------------------------------------

labeled_dataset <- read_csv("labeled_dataset.csv")
handcoding_dataset <- read_rds("Data/handcoding_dataset.rds")
paragraphs_cleaned <- read_rds("Data/paragraphs_cleaned.rds")

# clean labeled dataset --------------------------------------------------

# since I didn't label all the paragraphs in the handcoding dataset,
# I need to do some preprocessing to get the final labeled dataset

labeled_dataset_cleaned <-
  # join paragraphs_cleaned because in the first version i forgot to include the original paragraph
  inner_join(
    labeled_dataset,
    paragraphs_cleaned,
    by = join_by(paragraph_id)
  ) |>
  mutate(
    sentiment = case_when(
      sentiment == "non_climate" ~ FALSE,
      sentiment == "non_classifiable" ~ FALSE,
      sentiment == "climate" ~ TRUE
    ),
    WordCount = str_count(paragraph, "\\S+")
  ) |>
  rename(
    climate = sentiment,
    LanguageOfText = LanguageOfText.x
  ) |>
  select(
    !c(
      annotation_id,
      annotator,
      created_at,
      id,
      lead_time,
      updated_at
    )
  )

write_rds(labeled_dataset_cleaned, "Data/labeled_dataset_cleaned.rds")

# # save training dataset for BERT Fine-Tuning
# training_data <- labeled_dataset_cleaned |>
# rename(
#     original_text = paragraph,
#     language = LanguageOfText
#   ) |>
#   select(paragraph_id, original_text, final_climate, language)

# write_csv(training_data, "BERT_Finetuning/training_data.csv")

# transcripts_cleaned <- readRDS("Data/transcripts_cleaned.rds")
# transcripts_classified <- left_join(transcripts_cleaned, training_dataset, by = join_by(paragraph))

# plot data ------------------------------------------------------------------

# plot distribution of text lengths
plot_word_count_distribution(
  data = labeled_dataset_cleaned,
  title = "Distribution of Paragraph Lengths in Labeled Paragraphs",
  output_path = "Outputs/labeled_paragraphs_text_length_distribution.png"
)

# plot distribution of languages
plot_categorical_distribution(
  data = labeled_dataset_cleaned,
  group_col = "LanguageOfText",
  title = "Distribution of Languages in Labeled Paragraphs",
  x_label = "Language of Paragraphs",
  output_path = "Outputs/labeled_paragraphs_language_distribution.png"
)

# plot business tag distribution
plot_categorical_distribution(
  data = labeled_dataset_cleaned,
  group_col = "business_tag_climate",
  x_label = "Energy, Transport or Environment Related Business",
  title = "Distribution of Business Tags in Labeled Paragraphs",
  output_path = "Outputs/labeled_paragraphs_business_tags_distribution.png"
)

# plot distribution of final climate labels
plot_categorical_distribution(
  data = labeled_dataset_cleaned,
  group_col = "climate",
  x_label = "Class",
  title = "Distribution of Climate Related Paragraphs in Labeled Paragraphs",
  output_path = "Outputs/labeled_paragraphs_class_distribution.png"
)