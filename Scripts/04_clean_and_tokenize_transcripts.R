# libraries --------------------------------------------------------------

library(tidyverse)
library(tidytext)
source("Scripts/00_functions.R")

# import data ------------------------------------------------------------

transcripts <- read_rds("Data/transcripts.rds")
businesses_cleaned <- read_rds("Data/businesses_cleaned.rds")
subjects <- read_rds("Data/subjects.rds")

# filter transcripts -----------------------------------------------------

# sessions <- get_data(
#   "Session",
#   StartDate = c(">2015-01-01"),
#   Language = "DE"
# )

# # Gibt es Vorbesprechungen in Kommissionen?
# transcripts |> count(CouncilId, CouncilName) # beziehen sich auf den Sprecher
# transcripts |> count(MeetingCouncilAbbreviation)
# # alle Meetings sind entweder Nationalrat, Ständerat oder Vereinigte Bundesversammlung

# # check languages
# transcripts |> count(LanguageOfText) |> arrange(n)
# transcripts |> filter(is.na(LanguageOfText)) |> pull(Text) |> sample(10)
# transcripts |> filter(!is.na(LanguageOfText)) |> pull(Text) |> sample(10)
# # italienische Texte müssen sicher raus
# # aus den Samples ist nicht erkennbar, warum es NAs gibt

# # check speaker functions
# transcripts |> count(SpeakerFunction) |> arrange(n)
# transcripts |> filter(is.na(SpeakerFunction)) |> pull(Text) |> sample(10)
# # wo kein Parlamentarier zugewiesen machen kann, ist es meist eine Moderation
# # oder ein technischer Hinweis -> kann raus
# transcripts |> filter(SpeakerFunction == "P-M") |> pull(Text) |> sample(10)
# # bei P, VP, etc. bin ich mir nicht sicher ob relevant -> erstmal behalten

# # check Votes
# transcripts |> filter(!is.na(VoteBusinessNumber)) |> pull(Text) |> sample(10) # kann raus

# # check VS tags
# vorsitzender <- transcripts |>
#   filter(
#     str_detect(Text, "\\[VS]")
#   )
# # diese Transkript sind für die Analyse des Inhalts nicht von Interesse
# # diese werden daher aus dem Datensatz entfernt

# italienisch <- transcripts |>
#   filter(LanguageOfText == "IT")
# keine italienischen Texte, da nur ca. 1% der Texte

transcripts_filtered <- transcripts |>
  filter(
    IdSession >= 5002, # Frühjahrssession 2016
    IdSession <= 5210, # Herbstsession 2025
    LanguageOfText != "IT", # keine italienischen Texte
    is.na(VoteBusinessNumber), # keine Abstimmungen
    !is.na(SpeakerFunction), # keine Moderationen oder technische Hinweise
    !str_detect(Text, "\\[VS]") # remove paragraphs with VS (Vorsitzender) tags
  )

# session_statistics <- transcripts_filtered |>
#   group_by(IdSession) |>
#   summarise(
#     number_of_meetings = n_distinct(MeetingVerbalixOid),
#     number_of_protocols = n_distinct(ID),
#     number_of_businesses = n_distinct(IdSubject)
#   )

# meeting_statistics <- transcripts_filtered |>
#   group_by(MeetingVerbalixOid) |>
#   summarise(
#     number_of_protocols = n_distinct(ID),
#     number_of_businesses = n_distinct(IdSubject)
#   )

# saveRDS(transcripts_filtered, "Data/transcripts_filtered.rds")

# tokenize transcripts ---------------------------------------------------

# transcripts_filtered <- read_rds("Data/transcripts_filtered.rds")

paragraphs_raw <- transcripts_filtered |>
  mutate(
    paragraph = str_extract_all(Text, "(?<=<p>)(.*?)(?=</p>)")
  ) |>
  unnest(paragraph)

# saveRDS(transcripts_tokenized, "Data/transcripts_tokenized.rds")

# clean transcripts -----------------------------------------------------------

# transcripts_tokenized <- read_rds("Data/transcripts_tokenized.rds")

businesses_climate <- businesses_cleaned |>
  filter(business_tag_climate == TRUE) |>
  distinct(BusinessShortNumber) |>
  pull(BusinessShortNumber)

subjects_climate <- subjects |>
  filter(BusinessShortNumber %in% businesses_climate) |>
  distinct(IdSubject) |>
  pull(IdSubject)

# short_paragraphs <- transcripts_tokenized |>
#   filter(nchar(paragraph) < 50)

# italics <- transcripts_tokenized |>
#   filter(str_detect(paragraph, "<i>")) # remove the italics tags

# VS_tags <- transcripts_tokenized |>
#   filter(str_detect(paragraph, "\\[VS]")) # remove those parapgraphs

# GZ_tags <- transcripts_tokenized |>
#   filter(str_detect(paragraph, "\\[GZ]")) # handelt sich um eine Zäsur (langer Unterbruch) -> tags entfernen

paragraphs_cleaned <- paragraphs_raw |>
  group_by(ID) |>
  mutate(
    paragraph_id = paste0(ID, "-", row_number())
  ) |>
  ungroup() |>
  rename(transcript_id = ID) |>
  mutate(
    paragraph = paragraph |>
      str_replace_all("\\[PAGE \\d+\\]", "") |> # remove pagination
      str_replace_all("<[^>]+>", "") |> # remove HTML tags for italics and bold etc.
      str_replace_all("\\[GZ]", "") |> # remove [GZ] tags (Grosse Zäsur)
      str_squish(),
    business_tag_climate = if_else(
      IdSubject %in% subjects_climate,
      TRUE,
      FALSE
    ),
    WordCount = str_count(paragraph, "\\S+"),
    TextLength = nchar(paragraph),
    MeetingId = as.character(MeetingVerbalixOid),
    CouncilId = as.character(CouncilId),
    CantonId = as.character(CantonId)
  ) |>
  select(
    !c(
      VoteId,
      VoteBusinessNumber,
      VoteBusinessShortNumber,
      VoteBusinessTitle,
      Type,
      SortOrder
    )
  )

write_rds(paragraphs_cleaned, "Data/paragraphs_cleaned.rds")

# plot data ------------------------------------------------------------------

# transcripts_cleaned <- read_rds("Data/transcripts_cleaned.rds")
# transcripts_cleaned |> pull(paragraph) |> sample(10)

# plot paragraph length distribution
plot_word_count_distribution(
  data = paragraphs_cleaned,
  title = "Distribution of Paragraph Lengths in Complete Dataset",
  output_path = "Outputs/paragraphs_cleaned_text_length_distribution.png"
)

# plot language distribution
plot_categorical_distribution(
  data = paragraphs_cleaned,
  group_col = "LanguageOfText",
  x_label = "Language of Paragraphs",
  title = "Distribution of Languages in Complete Dataset",
  output_path = "Outputs/paragraphs_cleaned_language_distribution.png"
)

# plot business tag distribution
plot_categorical_distribution(
  data = paragraphs_cleaned,
  group_col = "business_tag_climate",
  x_label = "Energy, Transport or Environment Related Business",
  title = "Distribution of Business Tags in Complete Dataset",
  output_path = "Outputs/paragraphs_cleaned_business_tags_distribution.png"
)
