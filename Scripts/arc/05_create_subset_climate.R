# Description ------------------------------------------------------------

# The Goal is to create a sub-dataset that has with a high probability
# transcripts related to climate topics.
# For this we are using Tags

# libraries --------------------------------------------------------------

library(tidyverse)

# Filter businesses ------------------------------------------------------------

businesses_cleaned <- readRDS("Data/businesses_cleaned.rds")

businesses_long <- businesses_cleaned |>
  separate_rows(Tags, TagNames, sep = "\\|")

# analzise tags
businesses_long |>
  count(Tags, TagNames) |>
  print(n = Inf)

businesses_climate <- businesses_long |>
  filter(Tags %in% c(52, 48, 66)) |> # most relevant tags for climate change
  group_by(BusinessShortNumber) |>
  mutate(
    Tags = str_c(Tags, collapse = "|"),
    TagNames = str_c(TagNames, collapse = "|")
  ) |>
  ungroup()

saveRDS(businesses_climate, "Data/businesses_climate.rds")

# plot data --------------------------------------------------------------

subjects <- readRDS("Data/subjects.rds")
businesses_cleaned <- readRDS("Data/businesses_cleaned.rds")

joint_table <- left_join(transcripts_cleaned, subjects, by = join_by(IdSubject))
joint_table2 <- left_join(
  joint_table,
  businesses_cleaned,
  by = join_by(BusinessShortNumber)
)

df <- joint_table2 |>
  mutate(
    energy = str_detect(Tags, "66"),
    transport = str_detect(Tags, "48"),
    environment = str_detect(Tags, "52"),
    topic = case_when(
      energy ~ "energy",
      transport ~ "transport",
      environment ~ "environment",
      TRUE ~ "other"
    )
  )

ggplot(df, aes(x = topic)) +
  geom_bar()
