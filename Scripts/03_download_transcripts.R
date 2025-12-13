# libraries --------------------------------------------------------------

library(tidyverse)
library(swissparl)
library(furrr)
library(future)

# set up parallel --------------------------------------------------------

# set up parallel processing with background R sessions
# adjust number of workers as needed
# default: 6 for subject_business and 4 for transcripts
plan(multisession, workers = 4)

# get subject businesses --------------------------------------------------------

businesses_cleaned <- readRDS("Data/businesses_cleaned.rds")

get_subjects <- function(bsn, retries = 5, wait = 3) {
  folder <- "Data/subjects"
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
  businesses_cleaned$BusinessShortNumber,
  get_subjects,
  .progress = TRUE
)

# WAIT UNTIL ALL JOBS ARE FINISHED
value(fut)

subjects <- map_dfr(
  list.files("Data/subjects", full.names = TRUE),
  readRDS
)
saveRDS(subjects, "Data/subjects.rds")

# check results in SubjectBusiness ---------------------------------------

subjects <- readRDS("Data/subjects.rds")

businesses_cleaned |> distinct(BusinessShortNumber)
subjects |> distinct(BusinessShortNumber)
# some businesses have no entries in SubjectBusiness
# this is because some businesses have no discussion in the councils

# show be which businesses are missing
missing_bsn <- setdiff(
  businesses_cleaned$BusinessShortNumber,
  subjects$BusinessShortNumber
)
paste(length(missing_bsn), "businesses have no entries in SubjectBusiness.")

# # check whether businesses really have no entries in SubjectBusiness
# get_data(
#   table = "SubjectBusiness",
#   BusinessShortNumber = missing_bsn$BusinessShortNumber[1],
#   Language = "DE"
# )

# businesses_lost <- businesses_cleaned |> filter(BusinessShortNumber %in% missing_bsn)
# businesses_lost |> count(BusinessType, BusinessTypeName)

# # analyze businesses
# business_names <- subjects |> distinct(BusinessShortNumber, Title)

# get the transcripts ----------------------------------------------------

subjects <- readRDS("Data/subjects_2015_2025.rds")

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
#   subjects$IdSubject,
#   get_transcripts,
#   .progress = TRUE
# )

fut <- future_walk(
  subjects$IdSubject,
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
# subjects |> distinct(IdSubject)
# transcripts |> distinct(IdSubject)
