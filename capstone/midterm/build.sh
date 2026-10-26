#!/bin/sh
# Builds the Midterm PDF: the paper (work product) followed by the process documentation.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
for d in paper process_doc; do
  (cd "$HERE/../$d" && pdflatex -interaction=nonstopmode main >/dev/null && biber main >/dev/null && pdflatex -interaction=nonstopmode main >/dev/null && pdflatex -interaction=nonstopmode main >/dev/null)
done
pdfunite "$HERE/../paper/main.pdf" "$HERE/../process_doc/main.pdf" "$HERE/Weissman_Midterm_Deliverable.pdf"
echo "built $HERE/Weissman_Midterm_Deliverable.pdf"
