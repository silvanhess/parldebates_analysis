# libraries --------------------------------------------------------------

library(tidyverse)
library(swissparl)
library(tidytext)

# inspect transcripts table ----------------------------------------------

transcripts <- readRDS("Data/transcripts.rds")

sessions <- get_data(
  "Session",
  StartDate = c(">2015-01-01"),
  Language = "DE"
)

# Gibt es Vorbesprechungen in Kommissionen?
transcripts |> count(CouncilId, CouncilName) # beziehen sich auf den Sprecher
transcripts |> count(MeetingCouncilAbbreviation)
# alle Meetings sind entweder Nationalrat, Ständerat oder Vereinigte Bundesversammlung

# check languages
transcripts |> count(LanguageOfText) |> arrange(n)
transcripts |> filter(is.na(LanguageOfText)) |> pull(Text) |> sample(10)
transcripts |> filter(!is.na(LanguageOfText)) |> pull(Text) |> sample(10)
# italienische Texte müssen sicher raus
# aus den Samples ist nicht erkennbar, warum es NAs gibt

# check speaker functions
transcripts |> count(SpeakerFunction) |> arrange(n)
transcripts |> filter(is.na(SpeakerFunction)) |> pull(Text) |> sample(10)
# wo kein Parlamentarier zugewiesen machen kann, ist es meist eine Moderation
# oder ein technischer Hinweis -> kann raus
transcripts |> filter(SpeakerFunction == "P-M") |> pull(Text) |> sample(10)
# bei P, VP, etc. bin ich mir nicht sicher ob relevant -> erstmal behalten

# check Votes
transcripts |> filter(!is.na(VoteBusinessNumber)) |> pull(Text) |> sample(10) # kann raus

# filter transcripts -----------------------------------------------------

transcripts <- readRDS("Data/transcripts.rds")

transcripts_filtered <- transcripts |>
  filter(
    IdSession >= 5002, # Frühjahrssession 2016
    IdSession <= 5210, # Herbstsession 2025
    LanguageOfText != "IT",
    is.na(VoteBusinessNumber),
    !is.na(SpeakerFunction)
    # SpeakerFunction %in% c("Mit-F", "Mit-M")
  )

saveRDS(transcripts_filtered, "Data/transcripts_filtered.rds")

# filtered data inspection -------------------------------------------------

session_statistics <- transcripts_filtered |>
  group_by(IdSession) |>
  summarise(
    number_of_meetings = n_distinct(MeetingVerbalixOid),
    number_of_protocols = n_distinct(ID),
    number_of_businesses = n_distinct(IdSubject)
  )

meeting_statistics <- transcripts_filtered |>
  group_by(MeetingVerbalixOid) |>
  summarise(
    number_of_protocols = n_distinct(ID),
    number_of_businesses = n_distinct(IdSubject)
  )

# Tokenization and cleaning -----------------------------------------------

transcripts_filtered <- readRDS("Data/transcripts_filtered.rds")

transcripts_cleaned <- transcripts_filtered |>
  mutate(
    paragraph = str_extract_all(Text, "(?<=<p>)(.*?)(?=</p>)")
  ) |>
  unnest(paragraph) |>
  mutate(
    paragraph = paragraph |>
      str_replace_all("\\[PAGE \\d+\\]", "") |> # pagination entfernen
      str_squish()
  )

saveRDS(transcripts_cleaned, "Data/transcripts_cleaned.rds")

# insepct data ------------------------------------------------------------------

transcripts_cleaned <- readRDS("Data/transcripts_cleaned.rds")

transcripts_cleaned |> pull(paragraph) |> sample(10)

# plot the data
df <- transcripts_cleaned |>
  mutate(
    TextLength = nchar(paragraph)
  )

ggplot(df, aes(x = TextLength)) +
  geom_histogram() +
  xlim(0, 2000)
ggsave("Outputs/text_length_distribution.png")
