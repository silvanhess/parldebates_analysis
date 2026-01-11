# libraries --------------------------------------------------------------

library(tidyverse)

# import data ------------------------------------------------------------

paragraphs_cleaned <- read_rds("Data/paragraphs_cleaned.rds")

# create dataset for labeling --------------------------------------------

# create a balanced dataset for labeling
# for the initial training of the classifier we want to have
# 1000 paragraphs labeled with an even distribution between french and german
# also: we want to have both climate related and not climate related paragraphs
# to achieve this, we will sample more paragraphs from businesses that have a
# higher chance of having climate related paragraphs
# final dataset should have:

# 50/50 distribution between french and german
# 70/30 distribution between Climate Businesses and not Climate Businesses

# groups_before <- transcripts_cleaned |>
#   count(ClimateBusiness, LanguageOfText) |>
#   mutate(percentage = n / sum(n) * 100)

set.seed(1234)
handcoding_dataset <- paragraphs_cleaned |>
  mutate(
    cb_weight = if_else(business_tag_climate == TRUE, 10, 1),
    lang_weight = if_else(LanguageOfText == "FR", 3, 1),
    weight = cb_weight * lang_weight
  ) |>
  slice_sample(
    n = 1000,
    weight_by = weight
  )

# groups_after <- transcripts_sampled |>
#   count(ClimateBusiness, LanguageOfText) |>
#   mutate(percentage = n / sum(n) * 100)

write_rds(handcoding_dataset, "Data/handcoding_dataset.rds")

# plot data ------------------------------------------------------------------

# transcripts_sampled <- readRDS("Data/transcripts_sampled.rds")
# transcripts_cleaned |> pull(paragraph) |> sample(10)

# plot distribution of text lengths
plot_word_count_distribution(
  data = handcoding_dataset,
  title = "Distribution of Paragraph Lengths in Handcoding Dataset",
  output_path = "Outputs/handcoding_dataset_text_length_distribution.png"
)

# plot distribution of languages
plot_categorical_distribution(
  data = handcoding_dataset,
  group_col = "LanguageOfText",
  title = "Distribution of Languages in Handcoding Dataset",
  x_label = "Language of Paragraphs",
  output_path = "Outputs/handcoding_dataset_language_distribution.png")

# plot business tag distribution
plot_categorical_distribution(
  data = handcoding_dataset,
  group_col = "business_tag_climate",
  x_label = "Energy, Transport or Environment Related Business",
  title = "Distribution of Business Tags in Handcoding Dataset",
  output_path = "Outputs/handcoding_dataset_business_tags_distribution.png"
)