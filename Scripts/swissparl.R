# libraries --------------------------------------------------------------

library(tidyverse)
library(swissparl)
library(tidytext)
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
      token = "sentences"
    ) |>
    filter(nchar(CleanText) > 20) |> # filter short sentences
    select(
      !!id,
      !!date,
      !!oid,
      CleanText
    )
}

# check package content --------------------------------------------------

get_tables()
overview <- get_overview()
overview |> filter(variable == "IdSession")
get_variables("Subject")
sample_subjects <- get_glimpse("Subject", rows = 100)
get_variables("SubjectBusiness")
sample_subjectbusiness <- get_glimpse("SubjectBusiness", rows = 100)
get_variables("Business")
sample_business <- get_glimpse("Business", rows = 100)
get_variables("Session")
sample_sessions <- get_glimpse("Session", rows = 100)
get_variables("Meeting")
sample_meetings <- get_glimpse("Meeting", rows = 100)
get_variables("Transcript")
sample_transcripts <- get_glimpse("Transcript", rows = 100)

?get_data

# get transcripts from latest session -------------------------------------------

sessions <- get_data(
  "Session",
  StartDate = c(">2015-01-01"),
  Language = "DE"
)

transcripts_latest_session_path <- "Data/transcripts_latest_session.rds"

if (file.exists(transcripts_latest_session_path)) {
  transcripts_latest_session <- readRDS(transcripts_latest_session_path)
} else {
  transcripts_latest_session <- get_data(
    "Transcript",
    IdSession = "5210",
    Language = "DE"
  )
  saveRDS(transcripts_latest_session, transcripts_latest_session_path)
}

# get transcripts from specific businesses -------------------------------

businesses_energy <- get_data(
  table = "Business",
  Title = "~Klima",
  Language = "DE"
)

# businesses_energy_filtered <- businesses_energy |>
#   filter(SubmissionDate > as.Date("2025-01-01"))

subject_energy_list <- list()
for (i in seq_along(businesses_energy$BusinessShortNumber)) {
  subject_energy_list[[i]] <- get_data(
    table = "SubjectBusiness",
    BusinessShortNumber = businesses_energy$BusinessShortNumber[i],
    Language = "DE"
  )
}
subject_energy <- bind_rows(subject_energy_list)

transcripts_energy_list <- list()
for (i in seq_along(subject_energy$IdSubject)) {
  transcripts_energy_list[[i]] <- get_data(
    table = "Transcript",
    IdSubject = as.numeric(subject_energy$IdSubject),
    Language = "DE"
  )
}
transcripts_energy <- bind_rows(transcripts_energy_list)
saveRDS(transcripts_energy, "Data/transcripts_energy.rds")

# clean transcripts ------------------------------------------------------

transcripts_latest_session_clean <- clean_transcripts(
  transcripts_latest_session
)
transcripts_energy_clean <- clean_transcripts(transcripts_energy)

# analyze businesses latest session-----------------------------------------------------

businesses_latest_session_titles <- transcripts_latest_session |>
  distinct(VoteBusinessTitle) |>
  pull(VoteBusinessTitle)

businesses_latest_session_ids <- transcripts_latest_session |>
  distinct(VoteBusinessNumber) |>
  pull(VoteBusinessNumber)

# businesses_data <- get_data(
#   table = "Business",
#   ID %in% businesses_latest_session,
#   Language = "DE"
# ) # error object ID not found

# get businesses in a date range --------------------------------------------

# get business data for businesses that were submitted this year
businesses <- get_data(
  table = "Business",
  SubmissionDate = c(">2025-01-01", "<2025-11-19"),
  Language = "DE"
)

# unique(businesses$ID)

# get subjects for those businesses
# subjects <- get_data(
#   table = "SubjectBusiness",
#   BusinessShortNumber %in% businesses$BusinessShortNumber
# )
