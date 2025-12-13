# copy all files in folder Data/subject_businesses that are in the vector names to a new folder Data/subject_businesses_filtered

names <- paste0(businesses_filtered$BusinessShortNumber, ".rds")
src_folder <- "Data/subject_businesses"
dest_folder <- "Data/subject_businesses_filtered"
if (!dir.exists(dest_folder)) {
  dir.create(dest_folder)
}
file.copy(
  file.path(src_folder, names),
  file.path(dest_folder, names),
  overwrite = FALSE
)
