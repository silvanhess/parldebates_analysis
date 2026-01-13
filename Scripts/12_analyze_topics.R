# libraries --------------------------------------------------------------

library(tidyverse)

# import data ------------------------------------------------------------

documents_with_topics <- read_csv("Data/documents_with_topics.csv") |>
  mutate(
    transcript_id = as.character(transcript_id),
    topic = as.character(topic)
  )
topic_info <- read_csv("Data/topic_info.csv") |>
  rename(topic = Topic, topic_name = Name) |> 
  mutate(topic = as.character(topic))

paragraphs_classified_wide <- read_rds("Data/paragraphs_classified_wide.rds")

# plot political parties per topic ---------------------------------------

documents_with_topics_wide <- documents_with_topics |>
  left_join(topic_info, by = join_by(topic)) |>
  filter(topic != -1)

transcripts_topics_wide <- paragraphs_classified_wide |>
  inner_join(
    documents_with_topics_wide,
    by = join_by(transcript_id),
    relationship = "many-to-one"
  ) |> 
  drop_na(ParlGroupAbbreviation)

df_plot <- transcripts_topics_wide |> 
  filter(ParlGroupAbbreviation != "BD") |> 
  group_by(topic_name, ParlGroupAbbreviation) |>
  summarise(
    group_count = n(),
    .groups = 'drop_last'  # keeps topic grouping
  ) |>
  mutate(
    pct_of_topic = group_count / sum(group_count)
  ) |>
  drop_na()

# Create the legend data
party_legend <- transcripts_topics_wide |>
  mutate(ParlGroupName = case_when(
    ParlGroupAbbreviation == "M-E" ~ "Die Mitte-Fraktion / CVP-Fraktion",
    .default = ParlGroupName
  )) |> 
  distinct(ParlGroupAbbreviation, ParlGroupName)

# Join the legend to your data
df_with_names <- df_plot |>
  left_join(party_legend, by = "ParlGroupAbbreviation")

# Create plot with legend
ggplot(df_with_names, aes(x = ParlGroupAbbreviation, y = pct_of_topic, fill = ParlGroupName)) +
  geom_col() +
  facet_wrap(vars(topic_name)) +
  labs(
    x = "Parliamentary Group",
    y = "Percentage of Speeches",
    title = "Speeches per Party and Topic",
    fill = "Party Name"
  ) +
  theme_minimal() +
  theme(legend.position = "right")

ggsave("Outputs/topic_party_distribution.png")
