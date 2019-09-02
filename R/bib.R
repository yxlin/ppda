bib <- function() {
  sub("\\.bib$", "", system.file("bib", "ppda.bib", package = "ppda"))
}
