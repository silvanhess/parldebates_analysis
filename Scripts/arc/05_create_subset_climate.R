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

# # analzise tags
# businesses_long |>
#   count(Tags, TagNames) |>
#   print(n = Inf)

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
