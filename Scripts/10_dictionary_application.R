
# libraries --------------------------------------------------------------

# library(textdata)
library(tidyverse)
library(openxlsx)
library(tidytext)
library(stopwords)

# import data ------------------------------------------------------------

transcripts_cleaned <- readRDS("Data/transcripts_cleaned.rds")
french_dictionary_curated <- read.xlsx("french_dictionary_curated.xlsx") |> 
  rename(keyword_original = keyword_french)
german_dictionary_curated <- read.xlsx("german_dictionary_curated.xlsx") |> 
  rename(keyword_original = keyword_german)

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
german_stopwords <- stopwords::stopwords("de", source = "snowball")

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

# classify transcripts by counting matches with dictionary

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

# save as rds
write_rds(transcripts_classified, "Data/transcripts_classified.rds")
