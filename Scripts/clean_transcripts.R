# libraries --------------------------------------------------------------

library(tidyverse)
library(swissparl)
# ?swissparl
library(tidytext)
# ?tidytext
library(rlang)

# functions --------------------------------------------------------------

clean_transcripts <- function(
  df,
  text_col = "Text",
  id_col = "ID",
  date_col = "MeetingDate",
  oid_col = "MeetingVerbalixOid"
) {
  # Convert column names to symbols for tidy evaluation
  text <- rlang::sym(text_col)
  id <- rlang::sym(id_col)
  date <- rlang::sym(date_col)
  oid <- rlang::sym(oid_col)

  df |>
    mutate(
      CleanText = str_remove_all(!!text, "<.*?>"), # remove HTML
      CleanText = str_squish(CleanText) # normalize whitespace
    ) |>
    unnest_tokens(
      output = CleanText,
      input = CleanText,
      token = "paragraphs"
    ) #|>
  # filter(
  #   nchar(CleanText) > 100, # filter short sentences
  # ) |>
  # rename()
  # select(
  #   !!id,
  #   !!date,
  #   !!oid,
  #   CleanText
  # )
}

# clean transcripts ------------------------------------------------------

transcripts <- readRDS("Data/transcripts_filtered.rds")

transcripts_cleaned <- clean_transcripts(
  transcripts_filtered
)

# tests ------------------------------------------------------------------

df <- transcripts_cleaned |>
  mutate(TextLength = nchar(CleanText))

df$TextLength

# plot distribution of text lengths
ggplot(df, aes(x = TextLength)) +
  geom_histogram(binwidth = 1000) +
  # limit x scale to 10000
  xlim(0, 10000)
