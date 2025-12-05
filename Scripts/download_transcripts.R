# libraries --------------------------------------------------------------

library(tidyverse)
library(swissparl)
# ?swissparl
library(tidytext)
# ?tidytext
library(rlang)

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

# get businesses -------------------------------------------

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

year = 2015:2025
businesses_list <- list()
for (i in seq_along(year)) {
  businesses_list[[i]] <- get_data(
    "Business",
    SubmissionDate = c(
      paste(">", year[i], "-01-01", sep = ""),
      paste("<", year[i], "-12-31", sep = "")
    ),
    Language = "DE"
  )
}
businesses <- bind_rows(businesses_list)
saveRDS(businesses, "Data/businesses_2015_2025.rds")

# für alle Geschäfte, die im Parlament von 2015-2025 behandelt wurden,
# sind auch Geschäfte nötig, die bereits vor 2015 eingereicht wurden.

# get transcripts --------------------------------------------------------

subject_businesses_list <- list()
for (i in seq_along(businesses$BusinessShortNumber)) {
  subject_businesses_list[[i]] <- get_data(
    table = "SubjectBusiness",
    BusinessShortNumber = businesses$BusinessShortNumber[i],
    Language = "DE"
  )
  paste(
    "item",
    businesses$BusinessShortNumber[i],
    "of",
    length(businesses$BusinessShortNumber),
    "done"
  ) |>
    print()
}
subject_businesses <- bind_rows(subject_businesses_list)
saveRDS(subject_businesses, "Data/subject_businesses_2015_2025.rds")

transcripts_list <- list()
for (i in seq_along(subject_businesses$IdSubject)) {
  transcripts_list[[i]] <- get_data(
    table = "Transcript",
    IdSubject = subject_businesses$IdSubject[i],
    Language = "DE"
  )
  paste(
    "item",
    subject_businesses$IdSubject[i],
    "of",
    length(subject_businesses$IdSubject),
    "done"
  ) |>
    print()
}
transcripts <- bind_rows(transcripts_list)
saveRDS(transcripts, "Data/transcripts_energy.rds")

# get transcripts per business -------------------------------

businesses_energy <- get_data(
  table = "Business",
  Title = "~Klima",
  Language = "DE"
)

# TODO: Change Search string to english and include french and italian

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
