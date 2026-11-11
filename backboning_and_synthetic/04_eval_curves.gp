set terminal pdfcairo enhanced size 3,2.5 crop color solid rounded linewidth 1
set style fill solid

set key bottom right
set xlabel "FPR"
set ylabel "TPR"
f(x) = x
set yrange [0:1]

set output "roc_S.pdf"
plot "roc_S.tsv" u 1:3 w lines lc rgb "#e41a1c" lw 4 t "TDBB",\
"" u 1:($3-$8) w lines lc rgb "#e41a1c" lw 1 notitle,\
"" u 1:($3+$8) w lines lc rgb "#e41a1c" lw 1 notitle,\
"" u 1:5 w lines lc rgb "#377eb8" lw 4 t "NC",\
"" u 1:($5-$10) w lines lc rgb "#377eb8" lw 1 notitle,\
"" u 1:($5+$10) w lines lc rgb "#377eb8" lw 1 notitle,\
"" u 1:2 w lines lc rgb "#4daf4a" lw 4 t "DF",\
"" u 1:($2-$7) w lines lc rgb "#4daf4a" lw 1 notitle,\
"" u 1:($2+$7) w lines lc rgb "#4daf4a" lw 1 notitle,\
"" u 1:4 w lines lc rgb "#984ea3" lw 4 t "Naive",\
"" u 1:($4-$9) w lines lc rgb "#984ea3" lw 1 notitle,\
"" u 1:($4+$9) w lines lc rgb "#984ea3" lw 1 notitle,\
f(x) w lines lc rgb "#999999" lw 1 notitle

set output "roc_A.pdf"
plot "roc_A.tsv" u 1:3 w lines lc rgb "#e41a1c" lw 4 t "TDBB",\
"" u 1:($3-$8) w lines lc rgb "#e41a1c" lw 1 notitle,\
"" u 1:($3+$8) w lines lc rgb "#e41a1c" lw 1 notitle,\
"" u 1:5 w lines lc rgb "#377eb8" lw 4 t "NC",\
"" u 1:($5-$10) w lines lc rgb "#377eb8" lw 1 notitle,\
"" u 1:($5+$10) w lines lc rgb "#377eb8" lw 1 notitle,\
"" u 1:2 w lines lc rgb "#4daf4a" lw 4 t "DF",\
"" u 1:($2-$7) w lines lc rgb "#4daf4a" lw 1 notitle,\
"" u 1:($2+$7) w lines lc rgb "#4daf4a" lw 1 notitle,\
"" u 1:4 w lines lc rgb "#984ea3" lw 4 t "Naive",\
"" u 1:($4-$9) w lines lc rgb "#984ea3" lw 1 notitle,\
"" u 1:($4+$9) w lines lc rgb "#984ea3" lw 1 notitle,\
f(x) w lines lc rgb "#999999" lw 1 notitle

set output "roc_S_rnd.pdf"
plot "roc_S.tsv" u 1:3 w lines lc rgb "#cb181d" lw 4 t "Gravity",\
"" u 1:($3-$8) w lines lc rgb "#cb181d" lw 1 notitle,\
"" u 1:($3+$8) w lines lc rgb "#cb181d" lw 1 notitle,\
"" u 1:6 w lines lc rgb "#fcbba1" lw 4 t "Random",\
"" u 1:($6-$11) w lines lc rgb "#fcbba1" lw 1 notitle,\
"" u 1:($6+$11) w lines lc rgb "#fcbba1" lw 1 notitle,\
f(x) w lines lc rgb "#999999" lw 1 notitle

set output "roc_A_rnd.pdf"
plot "roc_A.tsv" u 1:3 w lines lc rgb "#cb181d" lw 4 t "Gravity",\
"" u 1:($3-$8) w lines lc rgb "#cb181d" lw 1 notitle,\
"" u 1:($3+$8) w lines lc rgb "#cb181d" lw 1 notitle,\
"" u 1:6 w lines lc rgb "#fcbba1" lw 4 t "Random",\
"" u 1:($6-$11) w lines lc rgb "#fcbba1" lw 1 notitle,\
"" u 1:($6+$11) w lines lc rgb "#fcbba1" lw 1 notitle,\
f(x) w lines lc rgb "#999999" lw 1 notitle

set key top right
set xlabel "Recall"
set ylabel "Precision"
set ytics 0.2

set output "pr_S.pdf"
plot "pr_S.tsv" u 1:3 w lines lc rgb "#e41a1c" lw 4 t "TDBB",\
"" u 1:($3-$8) w lines lc rgb "#e41a1c" lw 1 notitle,\
"" u 1:($3+$8) w lines lc rgb "#e41a1c" lw 1 notitle,\
"" u 1:5 w lines lc rgb "#377eb8" lw 4 t "NC",\
"" u 1:($5-$10) w lines lc rgb "#377eb8" lw 1 notitle,\
"" u 1:($5+$10) w lines lc rgb "#377eb8" lw 1 notitle,\
"" u 1:2 w lines lc rgb "#4daf4a" lw 4 t "DF",\
"" u 1:($2-$7) w lines lc rgb "#4daf4a" lw 1 notitle,\
"" u 1:($2+$7) w lines lc rgb "#4daf4a" lw 1 notitle,\
"" u 1:4 w lines lc rgb "#984ea3" lw 4 t "Naive",\
"" u 1:($4-$9) w lines lc rgb "#984ea3" lw 1 notitle,\
"" u 1:($4+$9) w lines lc rgb "#984ea3" lw 1 notitle

set output "pr_A.pdf"
plot "pr_A.tsv" u 1:3 w lines lc rgb "#e41a1c" lw 4 t "TDBB",\
"" u 1:($3-$8) w lines lc rgb "#e41a1c" lw 1 notitle,\
"" u 1:($3+$8) w lines lc rgb "#e41a1c" lw 1 notitle,\
"" u 1:5 w lines lc rgb "#377eb8" lw 4 t "NC",\
"" u 1:($5-$10) w lines lc rgb "#377eb8" lw 1 notitle,\
"" u 1:($5+$10) w lines lc rgb "#377eb8" lw 1 notitle,\
"" u 1:2 w lines lc rgb "#4daf4a" lw 4 t "DF",\
"" u 1:($2-$7) w lines lc rgb "#4daf4a" lw 1 notitle,\
"" u 1:($2+$7) w lines lc rgb "#4daf4a" lw 1 notitle,\
"" u 1:4 w lines lc rgb "#984ea3" lw 4 t "Naive",\
"" u 1:($4-$9) w lines lc rgb "#984ea3" lw 1 notitle,\
"" u 1:($4+$9) w lines lc rgb "#984ea3" lw 1 notitle

set output "pr_S_rnd.pdf"
plot "pr_S.tsv" u 1:3 w lines lc rgb "#cb181d" lw 4 t "Gravity",\
"" u 1:($3-$8) w lines lc rgb "#cb181d" lw 1 notitle,\
"" u 1:($3+$8) w lines lc rgb "#cb181d" lw 1 notitle,\
"" u 1:6 w lines lc rgb "#fcbba1" lw 4 t "Random",\
"" u 1:($6-$11) w lines lc rgb "#fcbba1" lw 1 notitle,\
"" u 1:($6+$11) w lines lc rgb "#fcbba1" lw 1 notitle

set output "pr_A_rnd.pdf"
plot "pr_A.tsv" u 1:3 w lines lc rgb "#cb181d" lw 4 t "Gravity",\
"" u 1:($3-$8) w lines lc rgb "#cb181d" lw 1 notitle,\
"" u 1:($3+$8) w lines lc rgb "#cb181d" lw 1 notitle,\
"" u 1:6 w lines lc rgb "#fcbba1" lw 4 t "Random",\
"" u 1:($6-$11) w lines lc rgb "#fcbba1" lw 1 notitle,\
"" u 1:($6+$11) w lines lc rgb "#fcbba1" lw 1 notitle
