#!/bin/bash

preamble() {
    echo '\begin{table}[!ht]'
    echo '\centering'

    chars=$(printf 'l%.0s' $(seq 0 "$1"))
    echo '\begin{tabular}{@{}c'"$chars"'@{}}'
}

postamble() {
    echo '\end{tabular}'
    echo '\end{table}'
}

all () {
    read first_line
    num_tabs=$(echo "${first_line}" | tr -dc "\t" | wc -c)
    preamble "$num_tabs"

    title=$(echo "$first_line" | sed 's/	/} \& \\multicolumn{1}{c}{/g')
    line_ending=' \\ \midrule'

    echo "       & \\multicolumn{1}{c}{${title}}${line_ending}"

    while read line; do
        new_line=$(echo "$line" | sed 's/	/ \& /g' | sed 's/%/\\%/g')
        echo "${new_line}${line_ending}"
    done

    postamble
}

all
