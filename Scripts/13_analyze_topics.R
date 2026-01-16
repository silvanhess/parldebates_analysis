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

# paragraphs_classified_wide <- read_rds("Data/paragraphs_classified_wide.rds")
paragraphs_climate <- read_rds("Data/paragraphs_climate.rds")

# plot political parties per topic ---------------------------------------

documents_with_topics_wide <- documents_with_topics |>
  left_join(topic_info, by = join_by(topic)) |>
  filter(
    topic != -1, # filter out outlier topic
    # topic %in% c(1, 2, 5, 6, 7, 11) # 1st cluster
    # topic %in% c(3, 4, 8, 9, 13) # 2nd cluster
    topic %in% c(0, 10, 12) # 3rd cluster
  ) 

paragraphs_topics_wide <- documents_with_topics_wide |>
  inner_join(
    paragraphs_climate,
    by = join_by(transcript_id),
    relationship = "many-to-many"
  ) |>
  drop_na(ParlGroupAbbreviation)

df_plot <- paragraphs_topics_wide |>
  filter(ParlGroupAbbreviation != "BD") |>
  group_by(topic_name, ParlGroupAbbreviation) |>
  summarise(
    group_count = n(),
    .groups = 'drop_last' # keeps topic grouping
  ) |>
  mutate(
    pct_of_topic = group_count / sum(group_count)
  ) |>
  drop_na()

# party order
party_order <- c("S", "G", "M-E", "GL", "RL", "V")

# Create the legend data
party_legend <- paragraphs_topics_wide |>
  mutate(
    ParlGroupName = case_when(
      ParlGroupAbbreviation == "M-E" ~ "Die Mitte-Fraktion / CVP-Fraktion",
      .default = ParlGroupName
    )
  ) |>
  distinct(ParlGroupAbbreviation, ParlGroupName)

# Define your color palette
party_colors <- c(
  "Fraktion der Schweizerischen Volkspartei" = "#00843D",
  "Sozialdemokratische Fraktion" = "#E4032E",
  "FDP-Liberale Fraktion" = "#0571C1",
  "Die Mitte-Fraktion / CVP-Fraktion" = "#FF8C00",
  "Grüne Fraktion" = "#84B547",
  "Grünliberale Fraktion" = "#650959ff"
)

# Join the legend to your data
df_with_names <- df_plot |>
  left_join(party_legend, by = "ParlGroupAbbreviation") |>
  mutate(
    ParlGroupAbbreviation = factor(ParlGroupAbbreviation, levels = party_order),
    ParlGroupName = factor(
      ParlGroupName,
      levels = unique(ParlGroupName[order(match(
        ParlGroupAbbreviation,
        party_order
      ))])
    )
  )

# Create plot with legend
ggplot(
  df_with_names,
  aes(x = ParlGroupAbbreviation, y = pct_of_topic, fill = ParlGroupName)
) +
  geom_col() +
  facet_wrap(vars(topic_name)) +
  scale_fill_manual(values = party_colors)
labs(
  x = "Parliamentary Group",
  y = "Percentage of Speeches",
  title = "Speeches per Party and Topic",
  fill = "Party Name"
) +
  theme_minimal() +
  theme(legend.position = "right")

ggsave("Outputs/topic_party_distribution.png")
