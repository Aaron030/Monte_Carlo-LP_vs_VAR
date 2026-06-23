#!/bin/sh
# Character/word counts for a thesis build's chapter content.
#
# Usage:  sh wordcount.sh <root-basename>      e.g.  sh wordcount.sh main_ext_1
# Run from the directory containing <root>.tex (this is latexmk's cwd).
# Wired to run after every successful compile via .latexmkrc ($success_cmd),
# so <root>.counts.txt refreshes on each PDF rebuild.
#
# Counts the content files this document \input's from chap/ or appendix/
# (so references, TOC, list of figures/tables, title page and AI declaration
# are excluded), and drops the figure "Note" blocks (the
# \begin{minipage}...\end{minipage} captions under pictures).
#
# Appendix files (any \input whose path contains "appendix") are still listed,
# but tagged "(excluded)" and NOT added to the TOTAL -- the total is body only.
#
# Characters include spaces; LaTeX markup and math are stripped by detex.

root="$1"
[ -n "$root" ] || { echo "usage: wordcount.sh <root>"; exit 1; }
main="${root}.tex"
[ -f "$main" ] || { echo "wordcount: $main not found in $(pwd)"; exit 0; }
out="${root}.counts.txt"

if ! command -v detex >/dev/null 2>&1; then
    echo "wordcount: detex not found on PATH; skipping ${out}" >&2
    exit 0
fi

# Content files \input by this document, in order, ignoring commented-out lines.
chaps=$(grep -vE '^[[:space:]]*%' "$main" \
        | grep -oE '\\input\{(chap|appendix)/[^}]+\}' \
        | sed -E 's/\\input\{(.+)\}/\1/')

# Count one file's text (figure-note minipages stripped); echoes "<chars> <words>".
count_file () {
    text=$(sed '/\\begin{minipage}/,/\\end{minipage}/d' "$1" \
           | detex 2>/dev/null | tr -s '[:space:]' ' ')
    c=$(printf '%s' "$text" | wc -m | tr -d ' ')
    w=$(printf '%s' "$text" | wc -w | tr -d ' ')
    echo "$c $w"
}

body_c=0; body_w=0
app_c=0;  app_w=0
rows=""
for f in $chaps; do
    case "$f" in *.tex) ;; *) f="${f}.tex" ;; esac
    [ -f "$f" ] || continue
    set -- $(count_file "$f")
    c="$1"; w="$2"
    name=$(basename "$f" .tex)
    case "$f" in
        *[Aa]ppendix*)
            app_c=$((app_c + c)); app_w=$((app_w + w))
            rows="${rows}$(printf '%-18s %14s %10s   %s' "$name" "$c" "$w" '(excluded)')
" ;;
        *)
            body_c=$((body_c + c)); body_w=$((body_w + w))
            rows="${rows}$(printf '%-18s %14s %10s' "$name" "$c" "$w")
" ;;
    esac
done

{
    echo "Character / word counts -- chapter content only (chars include spaces)."
    echo "Source: $main    Regenerated after each successful compile."
    echo "LaTeX markup and math excluded (via detex); figure Note blocks and"
    echo "auto-generated front/back matter (TOC, lists, references, title page) not counted."
    echo "Appendix files are listed but excluded from the TOTAL."
    echo
    printf '%-18s %14s %10s\n' "chapter" "chars(+spaces)" "words"
    printf '%-18s %14s %10s\n' "------------------" "--------------" "----------"
    printf '%s' "$rows"
    printf '%-18s %14s %10s\n' "------------------" "--------------" "----------"
    printf '%-18s %14s %10s\n' "TOTAL (excl. app.)" "$body_c" "$body_w"
    if [ "$app_c" -gt 0 ]; then
        printf '%-18s %14s %10s   %s\n' "appendix subtotal" "$app_c" "$app_w" "(excluded)"
    fi
} > "$out"

echo "wordcount: wrote $out (body $body_c chars / $body_w words; appendix excluded $app_c chars / $app_w words)"
