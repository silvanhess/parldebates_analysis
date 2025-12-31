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
  select(ID, paragraph, paragraph_translated.x, class) |>
  rename(paragraph_translated = paragraph_translated.x)

write_csv(training_dataset, "Data/training_dataset.csv")

# transcripts_cleaned <- readRDS("Data/transcripts_cleaned.rds")
# transcripts_classified <- left_join(transcripts_cleaned, training_dataset, by = join_by(paragraph))

# plot data ------------------------------------------------------------------

training_dataset <- read.csv("Data/training_dataset.csv")
# transcripts_cleaned |> pull(paragraph) |> sample(10)

transcripts_sampled <- readRDS("Data/transcripts_sampled.rds")

df <- inner_join(
  training_dataset,
  transcripts_sampled <- readRDS("Data/transcripts_sampled.rds"),
  by = join_by(ID, paragraph)
)

ggplot(df, aes(x = Textlength)) +
  geom_histogram() +
  xlim(0, 2000) +
  theme_minimal() +
  labs(
    x = "Paragraph Length (in characters)",
    y = "Number of Paragraphs",
    title = "Distribution of Paragraph Lengths in Training Dataset"
  )
ggsave("Outputs/training_dataset_text_length.png")

df_grouped <- df |>
  group_by(LanguageOfText) |>
  summarise(
    number_of_paragraphs = n(),
    pct_paragraphs = number_of_paragraphs / nrow(transcripts_sampled)
  )

ggplot(df_grouped, aes(x = LanguageOfText, y = pct_paragraphs)) +
  geom_col() +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    x = "Language of Paragraphs",
    y = "Percentage of Paragraphs",
    title = "Distribution of Languages in Training Dataset"
  ) +
  theme_minimal() +
  # label the columns with the number of paragraphs
  geom_text(
    aes(label = paste(number_of_paragraphs, "paragraphs")),
    vjust = -0.5
  )
ggsave("Outputs/training_dataset_language_distribution.png")

df_grouped <- df |>
  group_by(ClimateBusiness) |>
  summarise(
    number_of_paragraphs = n(),
    pct_paragraphs = number_of_paragraphs / nrow(transcripts_sampled)
  )

ggplot(df_grouped, aes(x = ClimateBusiness, y = pct_paragraphs)) +
  geom_col() +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    x = "Energy, Transport or Environment Related Paragraphs",
    y = "Percentage of Paragraphs",
    title = "Distribution of Climate Related Paragraphs in Training Dataset"
  ) +
  theme_minimal() +
  # label the columns with the number of paragraphs
  geom_text(
    aes(label = paste(number_of_paragraphs, "paragraphs")),
    vjust = -0.5
  )
ggsave("Outputs/training_dataset_topic_distribution.png")

df_grouped <- df |>
  group_by(class) |>
  summarise(
    number_of_paragraphs = n(),
    pct_paragraphs = number_of_paragraphs / nrow(transcripts_sampled)
  )

ggplot(df_grouped, aes(x = class, y = pct_paragraphs)) +
  geom_col() +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    x = "Class",
    y = "Percentage of Paragraphs",
    title = "Distribution of Climate Related Paragraphs in Training Dataset"
  ) +
  theme_minimal() +
  # label the columns with the number of paragraphs
  geom_text(
    aes(label = paste(number_of_paragraphs, "paragraphs")),
    vjust = -0.5
  )
ggsave("Outputs/training_dataset_class_distribution.png")
