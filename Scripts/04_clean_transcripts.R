# libraries --------------------------------------------------------------

library(tidyverse)
library(tidytext)

# filter transcripts -----------------------------------------------------

transcripts <- readRDS("Data/transcripts.rds")

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

saveRDS(transcripts_filtered, "Data/transcripts_filtered.rds")

# tokenize transcripts ---------------------------------------------------

transcripts_filtered <- readRDS("Data/transcripts_filtered.rds")

transcripts_tokenized <- transcripts_filtered |>
  mutate(
    paragraph = str_extract_all(Text, "(?<=<p>)(.*?)(?=</p>)")
  ) |>
  unnest(paragraph)

saveRDS(transcripts_tokenized, "Data/transcripts_tokenized.rds")

# clean transcripts -----------------------------------------------------------

transcripts_tokenized <- readRDS("Data/transcripts_tokenized.rds")

# climate related businesses
businesses_cleaned <- readRDS("Data/businesses_cleaned.rds")
subjects <- readRDS("Data/subjects.rds")

businesses_climate <- businesses_cleaned |>
  filter(ClimateBusiness == TRUE)

subjects_climate <- subjects |>
  filter(BusinessShortNumber %in% businesses_climate$BusinessShortNumber)

# short_paragraphs <- transcripts_tokenized |>
#   filter(nchar(paragraph) < 50)

# italics <- transcripts_tokenized |>
#   filter(str_detect(paragraph, "<i>")) # remove the italics tags

# VS_tags <- transcripts_tokenized |>
#   filter(str_detect(paragraph, "\\[VS]")) # remove those parapgraphs

# GZ_tags <- transcripts_tokenized |>
#   filter(str_detect(paragraph, "\\[GZ]")) # handelt sich um eine Zäsur (langer Unterbruch) -> tags entfernen

transcripts_cleaned <- transcripts_tokenized |>
  group_by(ID) |>
  mutate(
    paragraph_id = row_number()
  ) |>
  ungroup() |>
  mutate(
    paragraph = paragraph |>
      str_replace_all("\\[PAGE \\d+\\]", "") |> # remove pagination
      str_replace_all("<[^>]+>", "") |> # remove HTML tags for italics and bold etc.
      str_replace_all("\\[GZ]", "") |> # remove [GZ] tags (Grosse Zäsur)
      str_squish(),
    ClimateBusiness = if_else(
      IdSubject %in% subjects_climate$IdSubject,
      TRUE,
      FALSE
    ),
    ID = paste0(ID, "_", paragraph_id),
    Textlength = nchar(paragraph),
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

saveRDS(transcripts_cleaned, "Data/transcripts_cleaned.rds")

# plot data ------------------------------------------------------------------

transcripts_cleaned <- readRDS("Data/transcripts_cleaned.rds")
# transcripts_cleaned |> pull(paragraph) |> sample(10)

ggplot(transcripts_cleaned, aes(x = Textlength)) +
  geom_histogram() +
  xlim(0, 2000) +
  theme_minimal() +
  labs(
    x = "Paragraph Length (in characters)",
    y = "Number of Paragraphs",
    title = "Distribution of Paragraph Lengths in Complete Dataset"
  )
ggsave("Outputs/transcripts_cleaned_text_length.png")

df <- transcripts_cleaned |>
  group_by(LanguageOfText) |>
  summarise(
    number_of_paragraphs = n(),
    pct_paragraphs = number_of_paragraphs / nrow(transcripts_cleaned)
  )

ggplot(df, aes(x = LanguageOfText, y = pct_paragraphs)) +
  geom_col() +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    x = "Language of Paragraphs",
    y = "Percentage of Paragraphs",
    title = "Distribution of Languages in Complete Dataset"
  ) +
  theme_minimal() +
  # label the columns with the number of paragraphs
  geom_text(
    aes(label = paste(number_of_paragraphs, "paragraphs")),
    vjust = -0.5
  )
ggsave("Outputs/transcripts_cleaned_language_distribution.png")

df <- transcripts_cleaned |>
  group_by(ClimateBusiness) |>
  summarise(
    number_of_paragraphs = n(),
    pct_paragraphs = number_of_paragraphs / nrow(transcripts_cleaned)
  )

ggplot(df, aes(x = ClimateBusiness, y = pct_paragraphs)) +
  geom_col() +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    x = "Energy, Transport or Environment Related Paragraphs",
    y = "Percentage of Paragraphs",
    title = "Distribution of Climate Related Paragraphs in Complete Dataset"
  ) +
  theme_minimal() +
  # label the columns with the number of paragraphs
  geom_text(
    aes(label = paste(number_of_paragraphs, "paragraphs")),
    vjust = -0.5
  )
ggsave("Outputs/transcripts_cleaned_topic_distribution.png")
