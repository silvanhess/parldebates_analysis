# libraries --------------------------------------------------------------

library(tidyverse)

# import data ------------------------------------------------------------

labeled_dataset_raw <- read.csv("labeled_dataset.csv")

handcoding_dataset <- read.csv("Data/handcoding_dataset.csv")

labeled_dataset_cleaned <-
  # join handcoding dataset because in the first version i forgot to include the original paragraph
  inner_join(
    labeled_dataset_raw,
    handcoding_dataset,
    by = join_by(ID)
  ) |>
  mutate(
    sentiment = case_when(
      sentiment == "non_climate" ~ 0,
      sentiment == "non_classifiable" ~ 0,
      sentiment == "climate" ~ 1
    ),
    WordCount = str_count(paragraph, "\\S+")
  ) |>
  rename(
    paragraph_id = ID,
    original_text = paragraph,
    paragraph_translated = paragraph_translated.x,
    final_climate = sentiment,
    language = LanguageOfText.x
  ) |>
  select(
    paragraph_id,
    original_text,
    paragraph_translated,
    final_climate,
    language,
    WordCount
  )

saveRDS(labeled_dataset_cleaned, "Data/labeled_dataset_cleaned.rds")

# save training dataset
training_data <- labeled_dataset_cleaned |>
  select(paragraph_id,
    original_text,
    final_climate,
    language)

write_csv(training_data, "BERT_Finetuning/training_data.csv")

# transcripts_cleaned <- readRDS("Data/transcripts_cleaned.rds")
# transcripts_classified <- left_join(transcripts_cleaned, training_dataset, by = join_by(paragraph))

# plot data ------------------------------------------------------------------

labeled_dataset_cleaned <- readRDS("Data/labeled_dataset_cleaned.rds")

ggplot(labeled_dataset_cleaned, aes(x = WordCount)) +
  geom_histogram() +
  xlim(0, 300) +
  geom_vline(xintercept = 256, color = "red", linetype = "dashed") +
  annotate(
    "text",
    x = 260,
    y = 10,
    label = "Max Tokens",
    color = "red",
    hjust = 1.5
  ) +
  theme_minimal() +
  labs(
    x = "Paragraph Length (in words)",
    y = "Number of Paragraphs",
    title = "Distribution of Paragraph Lengths in Training Dataset"
  )
ggsave("Outputs/training_dataset_text_length.png")

df_grouped <- labeled_dataset_cleaned |>
  group_by(language) |>
  summarise(
    number_of_paragraphs = n(),
    pct_paragraphs = number_of_paragraphs / nrow(labeled_dataset_cleaned)
  )

ggplot(df_grouped, aes(x = language, y = pct_paragraphs)) +
  geom_col() +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    x = "Language of Paragraphs",
    y = "Percentage of Paragraphs",
    title = "Distribution of Languages in Training Dataset"
  ) +
  theme_minimal() +
  geom_text(
    aes(label = paste(number_of_paragraphs, "paragraphs")),
    vjust = -0.5
  )
ggsave("Outputs/training_dataset_language_distribution.png")

df_grouped <- labeled_dataset_cleaned |>
  group_by(final_climate) |>
  summarise(
    number_of_paragraphs = n(),
    pct_paragraphs = number_of_paragraphs / nrow(labeled_dataset_cleaned)
  )

ggplot(df_grouped, aes(x = final_climate, y = pct_paragraphs)) +
  geom_col() +
  scale_y_continuous(labels = scales::percent_format()) +
  scale_x_continuous(
    breaks = c(0, 1),
    labels = c("non_climate", "climate")
  ) +
  labs(
    x = "class",
    y = "Percentage of Paragraphs",
    title = "Distribution of Climate Related Paragraphs in Training Dataset"
  ) +
  theme_minimal() +
  geom_text(
    aes(label = paste(number_of_paragraphs, "paragraphs")),
    vjust = -0.5
  )
ggsave("Outputs/training_dataset_class_distribution.png")
