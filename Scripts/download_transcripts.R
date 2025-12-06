# libraries --------------------------------------------------------------

library(tidyverse)
library(swissparl)
# ?swissparl
library(tidytext)
# ?tidytext
library(rlang)
library(furrr)
library(future)

# set up parallel --------------------------------------------------------

# set up parallel processing with background R sessions
# adjust number of workers as needed
# default: 6 for subject_business and 4 for transcripts
plan(multisession, workers = 4)

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
get_variables("Session")
sample_sessions <- get_glimpse("Session", rows = 100)

?get_data

# tests ------------------------------------------------------------------

# business_25041 <- get_data(
#   "Business",
#   BusinessShortNumber = "25.041",
#   Language = "DE"
# )

# businesses_2014 <- get_data(
#   "Business",
#   # SubmissionCouncil = "1", # fails
#   # SubmissionSession = as.character(sessions$ID[1]), # fails
#   SubmissionDate = c(">2014-01-01", "<2014-12-31"),
#   # SubmissionDate = c(paste(">", year[1], "-01-01", sep = ""), paste("<", year[1], "-12-31", sep = "")),
#   Language = "DE"
# )

# get businesses -------------------------------------------

# für alle Geschäfte, die im Parlament von 2015-2025 behandelt wurden,
# sind auch Geschäfte nötig, die bereits vor 2015 eingereicht wurden.
# Daher werden hier alle Geschäfte von 2010-2025 heruntergeladen.

yr = 2010:2025

# with walk

get_businesses_year <- function(yr) {
  folder <- "Data/businesses"
  if (!dir.exists(folder)) {
    dir.create(folder)
  }

  out_file <- file.path(folder, paste0(yr, ".rds"))
  if (file.exists(out_file)) {
    return(NULL)
  } # skip existing files

  dt <- get_data(
    "Business",
    SubmissionDate = c(
      paste(">", yr, "-01-01", sep = ""),
      paste("<", yr, "-12-31", sep = "")
    ),
    Language = "DE"
  )

  saveRDS(dt, file.path(folder, paste0(yr, ".rds")))
}

walk(
  yr,
  get_businesses_year,
  .progress = TRUE
)

businesses <- map_dfr(list.files("Data/businesses", full.names = T), readRDS)
saveRDS(businesses, "Data/businesses_2015_2025.rds")

businesses <- readRDS("Data/businesses_2015_2025.rds")

# Filter businesses -----------------------------------------

# Nur Geschäftsarten, welche im Parlament behandelt wurden
# Aschliessen: Anfrage, Dringliche Anfrage, Petition, Interpellation, Fragestunde

# count business types
businesses |>
  count(BusinessType, BusinessTypeName) |>
  arrange(n) |>
  print(n = Inf)

# Nur Geschäfte im Status, wo sie im Parlament behandelt wurden

# show status
businesses_filtered |>
  count(BusinessStatus, BusinessStatusText) |>
  arrange(n) |>
  print(n = Inf)

# Apply filters
businesses_filtered <- businesses |>
  filter(
    !(BusinessType %in% c(18, 19, 10, 8, 14)),
    BusinessStatus %in% c(219, 222, 232, 229, 218, 215, 220)
  ) |>
  distinct(ID, .keep_all = TRUE)

# # check
# businesses_filtered |>
#   count(BusinessType, BusinessTypeName) |>
#   arrange(n) |>
#   print(n = Inf)

# businesses_filtered |>
#   count(BusinessStatus, BusinessStatusText) |>
#   arrange(n) |>
#   print(n = Inf)

saveRDS(businesses_filtered, "Data/businesses_filtered_2015_2025.rds")

# get subject businesses --------------------------------------------------------

businesses_filtered <- readRDS("Data/businesses_filtered_2015_2025.rds")

# with walk

get_subject_business <- function(bsn, retries = 5, wait = 3) {
  folder <- "Data/subject_businesses"
  if (!dir.exists(folder)) {
    dir.create(folder)
  }

  out_file <- file.path(folder, paste0(bsn, ".rds"))
  if (file.exists(out_file)) {
    return(NULL)
  } # skip existing files

  for (attempt in 1:retries) {
    result <- try(
      get_data(
        table = "SubjectBusiness",
        BusinessShortNumber = bsn,
        Language = "DE"
      ),
      silent = TRUE
    )

    if (!inherits(result, "try-error")) {
      saveRDS(result, out_file)
      # message("Downloaded ", bsn, " (attempt ", attempt, ")")
      return(TRUE)
    }

    # message("Error on ", bsn, " (attempt ", attempt, "): ", result)
    Sys.sleep(wait * attempt) # exponential backoff
  }

  # message("FAILED permanently: ", bsn)
  write(bsn, file = "failed_ids.txt", append = TRUE)
  return(FALSE)
}

# Start parallel jobs
fut <- future_walk(
  businesses_filtered$BusinessShortNumber,
  get_subject_business,
  .progress = TRUE
)

# WAIT UNTIL ALL JOBS ARE FINISHED
value(fut)

subject_businesses <- map_dfr(
  list.files("Data/subject_businesses", full.names = TRUE),
  readRDS
)
saveRDS(subject_businesses, "Data/subject_businesses_2015_2025.rds")

# check results in SubjectBusiness ---------------------------------------

# businesses_filtered |> distinct(BusinessShortNumber)
# subject_businesses |> distinct(BusinessShortNumber)
# # some businesses have no entries in SubjectBusiness
# # this is because some businesses have no discussion in the councils

# # # show be which businesses are missing
# missing_bsn <- setdiff(
#   businesses_filtered$BusinessShortNumber,
#   subject_businesses$BusinessShortNumber
# )
# length(missing_bsn)

# # check whether businesses really have no entries in SubjectBusiness
# get_data(
#   table = "SubjectBusiness",
#   BusinessShortNumber = missing_bsn$BusinessShortNumber[1],
#   Language = "DE"
# )

# businesses_lost <- businesses_filtered |> filter(BusinessShortNumber %in% missing_bsn)
# businesses_lost |> count(BusinessType, BusinessTypeName)

# get the transcripts ----------------------------------------------------

subject_businesses <- readRDS("Data/subject_businesses_2015_2025.rds")

# with walk

get_transcripts <- function(sbj, retries = 5, wait = 3) {
  folder <- "Data/transcripts"
  if (!dir.exists(folder)) {
    dir.create(folder)
  }

  out_file <- file.path(folder, paste0(sbj, ".rds"))
  if (file.exists(out_file)) {
    return(NULL)
  } # skip existing files

  for (attempt in 1:retries) {
    result <- try(
      get_data(
        table = "Transcript",
        IdSubject = as.double(sbj),
        Language = "DE"
      ),
      silent = TRUE
    )

    if (!inherits(result, "try-error")) {
      saveRDS(result, out_file)
      # message("Downloaded ", sbj, " (attempt ", attempt, ")")
      return(TRUE)
    }

    # message("Error on ", sbj, " (attempt ", attempt, "): ", result)
    Sys.sleep(wait * attempt) # exponential backoff
  }

  # message("FAILED permanently: ", sbj)
  write(sbj, file = "failed_ids.txt", append = TRUE)
  return(FALSE)
}

# walk(
#   subject_businesses$IdSubject,
#   get_transcripts,
#   .progress = TRUE
# )

fut <- future_walk(
  subject_businesses$IdSubject,
  get_transcripts,
  .progress = TRUE
)

value(fut)

transcripts <- map_dfr(
  list.files("Data/transcripts", full.names = TRUE),
  readRDS
)
saveRDS(transcripts, "Data/transcripts.rds")

# # check if all transcripts are there
# subject_businesses |> distinct(IdSubject)
# transcripts |> distinct(IdSubject)

# inspect transcripts table ----------------------------------------------

transcripts |> count(SpeakerFunction) |> arrange(n)
transcripts |> filter(is.na(SpeakerFunction)) |> pull(Text) |> sample(10)
transcripts |> filter(SpeakerFunction == "P-M") |> pull(Text) |> sample(10)

# Frühjahrsession 2015 und neuer
sessions <- get_data(
  "Session",
  StartDate = c(">2015-01-01"),
  Language = "DE"
)

# keine Kommissionssdebatten
transcripts |> count(CouncilId, CouncilName)
transcripts |> count(MeetingCouncilAbbreviation)

# kein italienisch
transcripts |> count(LanguageOfText) |> arrange(n)

# transcripts |> filter(is.na(LanguageOfText)) |> pull(Text) |> sample(10)
# transcripts |> filter(!is.na(LanguageOfText)) |> pull(Text) |> sample(10)

# filter transcripts -----------------------------------------------------

transcripts_filtered <- transcripts |>
  filter(
    IdSession >= 4917,
    # CouncilId %in% c(1, 2),
    # SpeakerFunction %in% c("Mit-F", "Mit-M", "P-F", "P-M"),
    LanguageOfText != "IT"
  )

saveRDS(transcripts_filtered, "Data/transcripts_filtered.rds")
