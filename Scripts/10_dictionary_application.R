
# libraries --------------------------------------------------------------

# library(textdata)
library(tidyverse)
library(openxlsx)
library(tidytext)
library(stopwords)

# import data ------------------------------------------------------------

transcripts_cleaned <- readRDS("Data/transcripts_cleaned.rds")
# french_dictionary_curated <- read.xlsx("french_dictionary_curated.xlsx")
german_dictionary_curated <- read.xlsx("german_dictionary_curated.xlsx")

# prepare data -----------------------------------------------------------

german_dictionary_curated_tidy <- german_dictionary_curated |> 
  select(keyword_german, category) |> 
  unnest_tokens(
    output = "word",
    input = "keyword_german",
    token = "words",
    drop = FALSE
  ) |> 
  mutate(
    word = str_to_lower(word),
    word = str_replace_all(word, "[^[:alnum:]'-]", ""),
    word = str_squish(word)
  ) |> 
  distinct()

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
  

# apply dictionary --------------------------------------------------------

transcripts_clasified <- transcripts_cleaned |> 
  left_join