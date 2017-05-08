bib <- function() {
  sub("\\.bib$", "", system.file("bib", "gpda.bib", package = "gpda"))
}
