# libraries --------------------------------------------------------------

library(tidyverse)
library(swissparl)
library(tidytext)

# inspect transcripts table ----------------------------------------------

# transcripts <- readRDS("Data/transcripts.rds")

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

# find climate related businesses ----------------------------------------

businesses_filtered <- readRDS("Data/businesses_filtered.rds")

search_strings <- c(
  "CO2",
  "Erneuerbare",
  "Heizung",
  "Klimafonds",
  "Solar",
  "Treibhaus",
  "Wärmepumpe",
  "Wasserkraft",
  "Windkraft",
  "Emissionshandel",
  "Energieeffizienz",
  "Energiespeicher",
  "Elektromobilität"
)

get_businesses <- function(search_strings) {
  folder <- "Data/businesses_climate"
  if (!dir.exists(folder)) {
    dir.create(folder)
  }

  out_file <- file.path(folder, paste0(search_strings, ".rds"))
  if (file.exists(out_file)) {
    return(NULL)
  } # skip existing files

  dt <- get_data(
    table = "Business",
    Title = paste0("~", search_strings),
    Language = "DE"
  )

  saveRDS(dt, file.path(folder, paste0(search_strings, ".rds")))
}

walk(
  search_strings,
  get_businesses,
  .progress = TRUE
)

businesses_climate <- map_dfr(
  list.files("Data/businesses_climate", full.names = T),
  readRDS
)
saveRDS(businesses_climate, "Data/businesses_climate.rds")

# Search climate businesses by filtering ---------------------------------

# search business titles with any of the search strings
businesses_climate <- businesses_filtered |>
  filter(str_detect(Title, paste(search_strings, collapse = "|")))

business_climate_old <- readRDS("Data/businesses_climate.rds")

# show diff between old and new
diff <- anti_join(
  businesses_climate,
  business_climate_old,
  by = "BusinessShortNumber"
)
# --> go with old version

# Tokenization and cleaning -----------------------------------------------

transcripts_filtered <- readRDS("Data/transcripts_filtered.rds")
subjects <- readRDS("Data/subjects.rds")
businesses_climate <- readRDS("Data/businesses_climate.rds")

subjects_climate <- subjects |>
  filter(BusinessShortNumber %in% businesses_climate$BusinessShortNumber)

transcripts_tokenized <- transcripts_filtered |>
  mutate(
    paragraph = str_extract_all(Text, "(?<=<p>)(.*?)(?=</p>)")
  ) |>
  unnest(paragraph)

# short_paragraphs <- transcripts_tokenized |>
#   filter(nchar(paragraph) < 50) # short paragraphs can be removed

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
  filter(
    !str_detect(paragraph, "\\[VS]"), # remove parapraphs with VS tags
    !nchar(paragraph) < 50 # remove short paragraphs
  ) |>
  mutate(
    paragraph = paragraph |>
      str_replace_all("\\[PAGE \\d+\\]", "") |> # remove pagination
      str_replace_all("<[^>]+>", "") |> # remove HTML tags for italics and bold etc.
      str_replace_all("\\[GZ]", "") |> # remove [GZ] tags
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

# insepct data ------------------------------------------------------------------

transcripts_cleaned <- readRDS("Data/transcripts_cleaned.rds")
# transcripts_cleaned |> pull(paragraph) |> sample(10)

ggplot(transcripts_cleaned, aes(x = Textlength)) +
  geom_histogram() +
  xlim(0, 2000)
ggsave("Outputs/text_length_distribution.png")

# join business info -----------------------------------------------------

transcripts_cleaned <- readRDS("Data/transcripts_cleaned.rds")
businesses_filtered <- readRDS("Data/businesses_filtered.rds")
subjects <- readRDS("Data/subjects.rds")

df <- businesses_filtered |>
  distinct(BusinessShortNumber, Title, BusinessTypeName) |>
  left_join(subjects) |>
  select(
    IdSubject,
    BusinessShortNumber,
    Title,
    BusinessTypeName,
    PublishedNotes
  )

transcripts_long <- left_join(transcripts_cleaned, df, by = join_by(IdSubject))

# number_of_businesses_per_transcript <- transcripts_long |>
#   count(ID) |>
#   arrange(desc(n))

# pivot_wider
transcripts_wide <- transcripts_long |>
  group_by(ID) |>
  mutate(business_number = row_number()) |>
  ungroup() |>
  pivot_wider(
    names_from = business_number,
    values_from = c(
      BusinessShortNumber,
      Title,
      BusinessTypeName,
      PublishedNotes
    ),
    names_sep = "_"
  )

# transcripts_wide |>
#   count(PublishedNotes_1) |>
#   arrange(desc(n))

# transcripts_wide |>
#   count(BusinessTypeName_1) |>
#   arrange(desc(n))

saveRDS(transcripts_wide, "Data/transcripts_wide.rds")
