# libraries --------------------------------------------------------------

library(tidyverse)
library(tidytext)
library(topicmodels)
library(tm)
library(slam)

# import data ------------------------------------------------------------

transcripts_classified <- readRDS("Data/transcripts_classified.rds")
businesses_cleaned <- readRDS("Data/businesses_cleaned.rds")
subjects <- readRDS("Data/subjects.rds")
transcripts_words_cleaned <- read_rds("Data/transcripts_words_cleaned.rds")

# join business info to transcripts --------------------------------------

# goal: make the business ID the document ID for topic modeling

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

# unique_businesses <- transcripts_subjects_businesses |>
#   distinct(BusinessShortNumber) |>
#   pull(BusinessShortNumber)

# businesses_debated <- businesses_cleaned |>
#   filter(BusinessShortNumber %in% unique_businesses)

# missing_businesses <- anti_join(
#   transcripts_subjects_businesses,
#   businesses_cleaned,
#   by = join_by(BusinessShortNumber)
# ) |>
#   distinct(BusinessShortNumber, Title)

# prepare data for topic modeling -----------------------------------------

df <- transcripts_subjects_businesses |>
  filter(climate_paragraph == TRUE) |>
  rename("ID" = "ID.x") |>
  select(ID, BusinessShortNumber, LanguageOfText) |>
  left_join(
    transcripts_words_cleaned,
    by = join_by(ID),
    relationship = "many-to-many"
  )

# count number of businesses
business_counts <- df |>
  distinct(BusinessShortNumber) |>
  count()

# count number of paragraphs
paragraph_counts <- df |>
  distinct(ID) |>
  count()

# print all the text of a random business
random_business <- sample(unique(df$BusinessShortNumber), 1)
df |>
  filter(BusinessShortNumber == random_business) |>
  arrange(ID) |>
  pull(word) |>
  paste(collapse = " ") |>
  cat()

# df <- transcripts_subjects_businesses |>
#   filter(climate_paragraph == TRUE) |>
#   select(BusinessShortNumber, LanguageOfText, paragraph.x) |>
#   # unnest the words
#   unnest_tokens(
#     output = "word",
#     input = "paragraph.x",
#     token = "words",
#     drop = TRUE
#   ) |>
#   # clean the words
#   mutate(
#     word = str_to_lower(word),
#     word = str_replace_all(word, "[^[:alnum:]'-]", ""),
#     word = str_squish(word)
#   )

# create document-term matrix --------------------------------------------

# german matrix
dtm_german <- df |>
  filter(LanguageOfText == "DE") |>
  count(BusinessShortNumber, word) |>
  cast_dtm(
    document = BusinessShortNumber,
    term = word,
    value = n
  )

# remove rare terms
# dtm_german_cleaned <- removeSparseTerms(dtm_german, 0.99)
term_doc_freq <- slam::col_sums(dtm_german > 0)
dtm_german_cleaned <- dtm_german[, term_doc_freq >= 5]

# french matrix
dtm_french <- df |>
  filter(LanguageOfText == "FR") |>
  count(BusinessShortNumber, word) |>
  cast_dtm(
    document = BusinessShortNumber,
    term = word,
    value = n
  )

# remove rare terms
# dtm_german_cleaned <- removeSparseTerms(dtm_german, 0.99)
term_doc_freq <- slam::col_sums(dtm_french > 0)
dtm_french_cleaned <- dtm_french[, term_doc_freq >= 5]

# apply topic modeling ----------------------------------------------------

start <- Sys.time()
set.seed(123)
lda_model_german <- LDA(dtm_german_cleaned, k = 8)
end <- Sys.time()
duration_german <- end - start
lda_topics_german <- tidy(lda_model_german, matrix = "beta")
lda_docs_german <- tidy(lda_model_german, matrix = "gamma")

start <- Sys.time()
set.seed(123)
lda_model_french <- LDA(dtm_french_cleaned, k = 8)
end <- Sys.time()
duration_french <- end - start
lda_topics_french <- tidy(lda_model_french, matrix = "beta")
lda_docs_french <- tidy(lda_model_german, matrix = "gamma")

# plot topics ------------------------------------------------------------

lda_topics_german |>
  group_by(topic) |>
  slice_max(beta, n = 10) |>
  mutate(term = reorder_within(term, beta, topic)) |>
  ggplot(aes(x = beta, y = term)) +
  geom_bar(stat = "identity") +
  facet_wrap(~topic, scales = "free_y") +
  scale_y_reordered()

lda_topics_french |>
  group_by(topic) |>
  slice_max(beta, n = 10) |>
  mutate(term = reorder_within(term, beta, topic)) |>
  ggplot(aes(x = beta, y = term)) +
  geom_bar(stat = "identity") +
  facet_wrap(~topic, scales = "free_y") +
  scale_y_reordered()

# show topics per document -----------------------------------------------

doc_classes_german <- lda_docs_german |> 
  group_by(document) |> 
  top_n(1) |> 
  ungroup()

doc_classes_french <- lda_docs_french |> 
  group_by(document) |> 
  top_n(1) |> 
  ungroup()

doc_classes_all <- full_join(
  doc_classes_german,
  doc_classes_french,
  by = join_by(document),
  suffix = c("_german", "_french")
)

