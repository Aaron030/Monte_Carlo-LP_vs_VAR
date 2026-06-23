#!/bin/sh
# Character/word counts for a thesis build's chapter content.
#
# Usage:  sh wordcount.sh <root-basename>      e.g.  sh wordcount.sh main_ext_1
# Run from the directory containing <root>.tex (this is latexmk's cwd).
# Wired to run after every successful compile via .latexmkrc ($success_cmd),
# so <root>.counts.txt refreshes on each PDF rebuild.
#
# Counts ONLY the chapters this document \input's (so references, TOC, list of
# figures/tables, title page and AI declaration are excluded), and drops the
# figure "Note" blocks (the \begin{minipage}...\end{minipage} captions under
# pictures). Characters include spaces; LaTeX markup and math stripped by detex.

root="$1"
[ -n "$root" ] || { echo "usage: wordcount.sh <root>"; exit 1; }
main="${root}.tex"
[ -f "$main" ] || { echo "wordcount: $main not found in $(pwd)"; exit 0; }
out="${root}.counts.txt"

if ! command -v detex >/dev/null 2>&1; then
    echo "wordcount: detex not found on PATH; skipping ${out}" >&2
    exit 0
fi

# Chapter files \input by this document, in order, ignoring commented-out lines.
chaps=$(grep -vE '^[[:space:]]*%' "$main" \
        | grep -oE '\\input\{chap/[^}]+\}' \
        | sed -E 's/\\input\{(.+)\}/\1/')

total_c=0
total_w=0
{
    echo "Character / word counts -- chapter content only (chars include spaces)."
    echo "Source: $main    Regenerated after each successful compile."
    echo "LaTeX markup and math excluded (via detex); figure Note blocks,"
    echo "auto-generated front/back matter (TOC, lists, references, title page) not counted."
    echo
    printf '%-18s %14s %10s\n' "chapter" "chars(+spaces)" "words"
    printf '%-18s %14s %10s\n' "------------------" "--------------" "----------"
} > "$out"

for f in $chaps; do
    case "$f" in *.tex) ;; *) f="${f}.tex" ;; esac
    [ -f "$f" ] || continue
    # Strip figure-note minipage blocks (notes under pictures) before counting.
    text=$(sed '/\\begin{minipage}/,/\\end{minipage}/d' "$f" \
           | detex 2>/dev/null | tr -s '[:space:]' ' ')
    c=$(printf '%s' "$text" | wc -m | tr -d ' ')
    w=$(printf '%s' "$text" | wc -w | tr -d ' ')
    total_c=$((total_c + c))
    total_w=$((total_w + w))
    printf '%-18s %14s %10s\n' "$(basename "$f" .tex)" "$c" "$w" >> "$out"
done

{
    printf '%-18s %14s %10s\n' "------------------" "--------------" "----------"
    printf '%-18s %14s %10s\n' "TOTAL" "$total_c" "$total_w"
} >> "$out"

echo "wordcount: wrote $out ($total_c chars incl. spaces, $total_w words)"
