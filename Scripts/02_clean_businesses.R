# libraries --------------------------------------------------------------

library(tidyverse)

# analyze businesses -----------------------------------------------------

# businesses <- readRDS("Data/businesses.rds")

# # analyze business types
# businesses |>
#   count(BusinessType, BusinessTypeName) |>
#   arrange(n) |>
#   print(n = Inf)

# # analyze business status
# businesses |>
#   count(BusinessStatus, BusinessStatusText) |>
#   arrange(n) |>
#   print(n = Inf)

# # analyze business tags
# businesses |>
#   separate_rows(Tags, TagNames, sep = "\\|") |>
#   count(Tags, TagNames) |>
#   print(n = Inf)

# # analyze Verkehr
# businesses |>
#   filter(str_detect(TagNames, "Verkehr")) |>
#   count(Title) |>
#   arrange(desc(n)) |>
#   print(n = Inf)

# Clean businesses -----------------------------------------

businesses <- readRDS("Data/businesses.rds")

businesses_cleaned <- businesses |>
  filter(
    !(BusinessType %in% c(18, 19, 10, 8, 14)), # Nur Geschäftsarten, welche im Parlament behandelt werden
    BusinessStatus %in% c(219, 222, 232, 229, 218, 215, 220) # Nur Geschäfte im Status, wo sie im Parlament behandelt wurden oder noch werden
  ) |>
  mutate(
    BusinessDetails_short = ifelse(is.na(Description), SubmittedText, Description), # facilitates later analysis of businesses
    BusinessDetails_long = paste(BusinessDetails_short, ReasonText),
    BusinessDetails_short_length = str_length(BusinessDetails_short),
    BusinessDetails_long_length = str_length(BusinessDetails_long),
    ClimateBusiness = str_detect(Tags, "52|48|66"), # Mark climate related businesses based on tags
  )

saveRDS(businesses_cleaned, "Data/businesses_cleaned.rds")

# analyze cleaned businesses ---------------------------------------------

businesses_cleaned <- readRDS("Data/businesses_cleaned.rds")

# # what tags are most common in Climate Businesses?
# businesses_cleaned |>
#   filter(ClimateBusiness == TRUE) |>
#   separate_rows(Tags, TagNames, sep = "\\|") |>
#   count(Tags, TagNames) |>
#   arrange(desc(n)) |>
#   print(n = Inf)

# analyze business details
businesses_cleaned |> 
  filter(ClimateBusiness == TRUE) |>
  slice_sample(n = 10) |>
  pull(BusinessDetails_long)

