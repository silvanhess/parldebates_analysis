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
plan(multisession, workers = 6)

# check package content --------------------------------------------------

# get_tables()
# overview <- get_overview()
# overview |> filter(variable == "IdSession")
# get_variables("Subject")
# sample_subjects <- get_glimpse("Subject", rows = 100)
# get_variables("SubjectBusiness")
# sample_subjectbusiness <- get_glimpse("SubjectBusiness", rows = 100)
# get_variables("Business")
# sample_business <- get_glimpse("Business", rows = 100)
# get_variables("Session")
# sample_sessions <- get_glimpse("Session", rows = 100)
# get_variables("Meeting")
# sample_meetings <- get_glimpse("Meeting", rows = 100)
# get_variables("Transcript")
# sample_transcripts <- get_glimpse("Transcript", rows = 100)
# get_variables("Session")
# sample_sessions <- get_glimpse("Session", rows = 100)

# ?get_data

# tests ------------------------------------------------------------------

# business_25041 <- get_data(
#   "Business",
#   BusinessShortNumber = "25.041",
#   Language = "DE"
# )

# sessions <- get_data(
#   "Session",
#   StartDate = c(">2015-01-01"),
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

# Nur Geschäfte, welche im Parlament behandelt wurden
# Aschliessen: Anfrage, Dringliche Anfrage, Petition, Interpellation, Fragestunde

# count business types
businesses |>
  count(BusinessType, BusinessTypeName) |>
  arrange(n) |>
  print(n = Inf)

# filter business types
businesses_filtered <- businesses |>
  filter(!(BusinessType %in% c(18, 19, 10, 8, 14)))

saveRDS(businesses_filtered, "Data/businesses_filtered_2015_2025.rds")

# get subject businesses --------------------------------------------------------

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
      message("Downloaded ", bsn, " (attempt ", attempt, ")")
      return(TRUE)
    }

    message("Error on ", bsn, " (attempt ", attempt, "): ", result)
    Sys.sleep(wait * attempt) # exponential backoff
  }

  message("FAILED permanently: ", bsn)
  write(bsn, file = "failed_ids.txt", append = TRUE)
  return(FALSE)
}

# Start parallel jobs
fut <- future_walk(
  businesses$BusinessShortNumber,
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

# get the transcripts ----------------------------------------------------

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
        IdSubject = sbj,
        Language = "DE"
      ),
      silent = TRUE
    )

    if (!inherits(result, "try-error")) {
      saveRDS(result, out_file)
      message("Downloaded ", sbj, " (attempt ", attempt, ")")
      return(TRUE)
    }

    message("Error on ", sbj, " (attempt ", attempt, "): ", result)
    Sys.sleep(wait * attempt) # exponential backoff
  }

  message("FAILED permanently: ", sbj)
  write(sbj, file = "failed_ids.txt", append = TRUE)
  return(FALSE)
}

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

# filter transcripts:
# only debates after 1.1.2025
# debates around climate change
