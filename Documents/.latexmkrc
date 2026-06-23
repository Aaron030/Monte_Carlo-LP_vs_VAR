# Project latexmk config (auto-read when latexmk runs in this directory).
#
# After every successful compile, regenerate <root>.counts.txt with the
# character/word counts of the chapters that document actually \input's.
# %R is the root filename base (e.g. main_ext_1), substituted by latexmk.
$success_cmd = 'sh wordcount.sh %R';
