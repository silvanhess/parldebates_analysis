# libraries --------------------------------------------------------------

# library(textdata)
library(tidyverse)
library(openxlsx)
library(tidytext)
library(stopwords)

# import data ------------------------------------------------------------

transcripts_cleaned <- read_rds("Data/transcripts_cleaned.rds")
french_dictionary_curated <- read.xlsx("french_dictionary_curated.xlsx") |>
  rename(keyword_original = keyword_french)
german_dictionary_curated <- read.xlsx("german_dictionary_curated.xlsx") |>
  rename(keyword_original = keyword_german)
businesses_cleaned <- read_rds("Data/businesses_cleaned.rds")
subjects <- read_rds("Data/subjects.rds")

# prepare data -----------------------------------------------------------

transcripts_words <- transcripts_cleaned |>
  select(ID, paragraph) |>
  unnest_tokens(
    output = "word",
    input = "paragraph",
    token = "words",
    drop = FALSE
  )

# get german stopwords from stopwords package
# add additional stopwords
german_stopwords <- c(
  stopwords::stopwords("de", source = "snowball"),
  "dass"
) |>
  sort()

# get french stopwords from stopwords package
french_stopwords <- stopwords::stopwords("fr", source = "snowball")

# put together stopwords
stopwords_all <- tibble(stopwords = c(german_stopwords, french_stopwords))

transcripts_words_cleaned <- transcripts_words |>
  mutate(
    word = str_to_lower(word),
    word = str_replace_all(word, "[^[:alnum:]'-]", ""),
    word = str_squish(word)
  ) |>
  anti_join(stopwords_all, by = join_by(word == stopwords))

# save as rds
write_rds(transcripts_words_cleaned, "Data/transcripts_words_cleaned.rds")

dictionary_all_cleaned <- german_dictionary_curated |>
  bind_rows(french_dictionary_curated) |>
  mutate(
    keyword_original = str_to_lower(keyword_original),
    keyword_original = str_replace_all(keyword_original, "[^[:alnum:]'-]", ""),
    keyword_original = str_squish(keyword_original)
  )

# save as rds
# write_rds(dictionary_all_cleaned, "Data/dictionary_all_cleaned.rds")

# apply dictionary --------------------------------------------------------

# classify transcript paragraphs by counting matches with dictionary

transcripts_classified <- transcripts_words_cleaned |>
  left_join(dictionary_all_cleaned, by = join_by(word == keyword_original)) |>
  group_by(ID, paragraph) |>
  summarise(
    keywords_found = paste(na.omit(unique(keyword_english)), collapse = ", "),
    n_keywords_found = sum(!is.na(keyword_english)),
    climate_paragraph = if_else(n_keywords_found > 0, TRUE, FALSE),
    .groups = "drop"
  ) |>
  left_join(transcripts_cleaned, by = join_by(ID))

# classify climate relevant businesses -----------------------------------

transcripts_subjects <- left_join(
  transcripts_classified,
  subjects,
  by = join_by(IdSubject),
  relationship = "many-to-many"
)

transcripts_subjects_businesses <- left_join(
  transcripts_subjects,
  businesses_cleaned,
  by = join_by(BusinessShortNumber),
  relationship = "many-to-one"
)

climate_businesses <- transcripts_subjects_businesses |>
  filter(climate_paragraph == TRUE) |>
  distinct(BusinessShortNumber) |>
  pull(BusinessShortNumber)

other_businesses <- transcripts_subjects_businesses |>
  filter(climate_paragraph == FALSE) |>
  distinct(BusinessShortNumber) |>
  pull(BusinessShortNumber)

# proportion of climate relevant businesses
proportion_climate_businesses <- length(climate_businesses) /
  length(other_businesses)

df_climate_wide <- transcripts_subjects_businesses |>
  mutate(
    # create a variable climate_business indicating whether the business contains climate relevant paragraphs
    climate_business = case_when(
      BusinessShortNumber %in% climate_businesses ~ TRUE,
      .default = FALSE
    )
  )

write_rds(df_climate_wide, "Data/df_climate_wide.rds")

# inspect business titles
df_climate_wide |>
  distinct(Title.x) |>
  pull(Title.x) |>
  sample(100)

# # inspect results ---------------------------------------------------------

# transcripts_classified <- read_rds("Data/transcripts_classified.rds")

# # check random paragraphs
# random_paragraphs <- sample(unique(transcripts_classified$ID), 10)
# transcripts_classified |>
#   filter(ID %in% random_paragraphs) |>
#   arrange(ID) |>
#   select(ID, paragraph.x, climate_paragraph, n_keywords_found, keywords_found) |>
#   print(n = 10)
