# Project latexmk config (auto-read when latexmk runs in this directory).
#
# Keep the working tree clean: all auto-generated files (.aux, .log, .bcf,
# .fls, .fdb_latexmk, .out, .toc, .lof, .lot, .run.xml, .synctex.gz, .bbl,
# .blg, ...) go into latex_logs/, while the final PDF stays in Documents/.
$aux_dir = 'latex_logs';

# After every successful compile, regenerate <root>.counts.txt with the
# character/word counts of the chapters that document actually \input's.
# %R is the root filename base (e.g. main_ext_1), substituted by latexmk.
$success_cmd = 'sh wordcount.sh %R';
