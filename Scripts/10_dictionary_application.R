# libraries --------------------------------------------------------------

# library(textdata)
library(tidyverse)
library(openxlsx)
library(tidytext)
library(stopwords)
source("Scripts/00_functions.R")

# import data ------------------------------------------------------------

paragraphs_cleaned <- read_rds("Data/paragraphs_cleaned.rds")
french_dictionary_curated <- read.xlsx("french_dictionary_curated.xlsx") |>
  rename(keyword_original = keyword_french)
german_dictionary_curated <- read.xlsx("german_dictionary_curated.xlsx") |>
  rename(keyword_original = keyword_german)
businesses_cleaned <- read_rds("Data/businesses_cleaned.rds")
subjects <- read_rds("Data/subjects.rds")

# prepare data -----------------------------------------------------------

paragraphs_words <- paragraphs_cleaned |>
  select(paragraph_id, paragraph) |>
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

paragraphs_words_cleaned <- paragraphs_words |>
  mutate(
    word = str_to_lower(word),
    word = str_replace_all(word, "[^[:alnum:]'-]", ""),
    word = str_squish(word)
  ) |>
  anti_join(stopwords_all, by = join_by(word == stopwords))

# save as rds
write_rds(paragraphs_words_cleaned, "Data/paragraphs_words_cleaned.rds")

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

paragraphs_classified <- paragraphs_words_cleaned |>
  left_join(dictionary_all_cleaned, by = join_by(word == keyword_original)) |>
  group_by(paragraph_id, paragraph) |>
  summarise(
    keywords_found = paste(na.omit(unique(keyword_english)), collapse = ", "),
    n_keywords_found = sum(!is.na(keyword_english)),
    climate_paragraph = if_else(n_keywords_found > 0, TRUE, FALSE),
    .groups = "drop"
  ) |>
  left_join(paragraphs_cleaned, by = join_by(paragraph_id))

# # insepect paragraphs
# transcripts_classified |>
#   filter(climate_paragraph == TRUE) |>
#   pull(paragraph.x) |>
#   sample(10)

# # check languages of climate relevant paragraphs
# transcripts_classified |>
#   filter(climate_paragraph == TRUE) |>
#   count(LanguageOfText)

# classify climate relevant businesses -----------------------------------

paragraphs_subjects <- left_join(
  paragraphs_classified,
  subjects,
  by = join_by(IdSubject),
  relationship = "many-to-many"
)

paragraphs_subjects_businesses <- left_join(
  paragraphs_subjects,
  businesses_cleaned,
  by = join_by(BusinessShortNumber),
  relationship = "many-to-one"
)

climate_businesses <- paragraphs_subjects_businesses |>
  filter(climate_paragraph == TRUE) |>
  distinct(BusinessShortNumber) |>
  pull(BusinessShortNumber)

# other_businesses <- transcripts_subjects_businesses |>
#   filter(climate_paragraph == FALSE) |>
#   distinct(BusinessShortNumber) |>
#   pull(BusinessShortNumber)

# # proportion of climate relevant businesses
# proportion_climate_businesses <- length(climate_businesses) /
#   length(other_businesses)

paragraphs_classified_wide <- paragraphs_subjects_businesses |>
  mutate(
    # create a variable climate_business indicating whether the business contains climate relevant paragraphs
    climate_business = case_when(
      BusinessShortNumber %in% climate_businesses ~ TRUE,
      .default = FALSE
    )
  ) |>
  rename(paragraph = paragraph.x)

write_rds(paragraphs_classified_wide, "Data/paragraphs_classified_wide.rds")

paragraphs_climate_wide <- paragraphs_classified_wide |>
  filter(climate_business == TRUE)

write_rds(paragraphs_climate_wide, "Data/paragraphs_climate_wide.rds")

# # inspect business titles
# transcript_climate_wide |>
#   distinct(Title.x) |>
#   pull(Title.x) |>
#   sample(100)

# plot results -----------------------------------------------------------

# plot language distribution
plot_categorical_distribution(
  paragraphs_climate_wide,
  group_col = "LanguageOfText",
  x_label = "Speech Language",
  title = "Language Distribution in Climate Relevant Businesses",
  output_path = "Outputs/paragraphs_climate_language_distribution.png"
) # this includes paragraphs not classified as climate relevant but belonging to a climate relevant business

# plot climate paragraph distribution
plot_categorical_distribution(
  paragraphs_classified_wide,
  group_col = "climate_paragraph",
  x_label = "Climate Relevant Paragraph",
  title = "Distribution of Climate Relevant Paragraphs",
  output_path = "Outputs/paragraphs_climate_class_distribution.png"
) # this includes only paragraphs directly classified as climate relevant

# plot paragraph distribution in climate relevant businesses
plot_categorical_distribution(
  paragraphs_classified_wide,
  group_col = "climate_business",
  x_label = "Climate Relevant Business",
  title = "Distribution of Climate Relevant Businesses",
  output_path = "Outputs/businesses_climate_paragraph_distribution.png"
) # this includes paragraphs not classified as climate relevant but belonging to a climate relevant business

# Aggragate to business level

businesses_climate <- paragraphs_classified_wide |>
  distinct(BusinessShortNumber, climate_business)

# plot climate business distribution
plot_categorical_distribution(
  businesses_climate,
  group_col = "climate_business",
  x_label = "Climate Relevant Business",
  label_suffix = "businesses",
  title = "Distribution of Climate Relevant Businesses",
  output_path = "Outputs/businesses_climate_class_distribution.png"
)
