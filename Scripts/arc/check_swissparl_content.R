# libraries --------------------------------------------------------------

library(tidyverse)
library(swissparl)

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

business_25041 <- get_data(
  "Business",
  BusinessShortNumber = "25.041",
  Language = "DE"
)

businesses_2014 <- get_data(
  "Business",
  # SubmissionCouncil = "1", # fails
  # SubmissionSession = as.character(sessions$ID[1]), # fails
  SubmissionDate = c(">2014-01-01", "<2014-12-31"),
  # SubmissionDate = c(paste(">", year[1], "-01-01", sep = ""), paste("<", year[1], "-12-31", sep = "")),
  Language = "DE"
)
