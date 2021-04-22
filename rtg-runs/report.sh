#!/usr/bin/env bash

export PYTHONPATH=~/work/isi/sacrebleu  #/home/07394/tgowda/repos/sacre-BLEU
MULTIBLEU=~/repos/mosesdecoder/scripts/generic/multi-bleu-detok.perl

LC="-lc"
function sacre_bleu {
    hyp=$1
    ref=$2
    lang=en-es

    if [[ -f $hyp ]]; then
        # ;s/<pad>//g
        score=$(cut -f1 $hyp | sed 's/<unk>//g' | python -m sacrebleu -m bleu -b -t $ref -l $lang $LC )
        echo $score
    else
        echo "NA-Hyp"
    fi
}

function multi_bleu {
    hyp=$1
    ref=$2
    if [[ -f $ref && -f $hyp ]]; then
        # ;s/<pad>//g
        score=$(cut -f1 $hyp | sed 's/<unk>//g' | $MULTIBLEU  $ref | sed 's/,//g' | cut -d ' ' -f3 )
        echo $score
    else
        if [[ ! -f $hyp ]]; then
            echo "NA-Hyp"
        else
            echo "NA-Ref"
        fi
    fi
}


function macro_f {
    hyp=$1
    ref=$2
    lang=en-es
    if [[ -f $hyp ]]; then
        # ;s/<pad>//g
        score=$(cut -f1 $hyp | sed 's/<unk>//g' | python -m sacrebleu -m macrof -w 2 -b -t $ref -l $lang $LC )
        echo $score
    else
        echo "NA-Hyp"
    fi
}

delim=','

#echo ${@}



#printf "Experiment${delim}SacreBLEU:${names_str}${delim}MultiBLEU:${names_str}\n"
printf "Experiment${delim}NT13_BLEU${delim}NT13_MacroF\n"
#exit 1

for d in ${@}; do
    for td in $d/test_*; do
    printf "$td"
        #for t in $names; do
        #    hyp_detok=${td}/$t.out.*detok
        #    ref=${td}/$t.ref
        #    bleu=$(sacre_bleu $hyp_detok $ref)
        #    printf "${delim}${bleu}"
        #done
    hyp_detok=$(echo ${td}/newstest2013.out.*detok)
    #echo $hyp_detok
    ref="wmt13"
    bleu=$(sacre_bleu $hyp_detok "$ref")
    printf "${delim}${bleu}"

    macrof=$(macro_f $hyp_detok "$ref")
    printf "${delim}${macrof}"

    printf "\n"
    done
done
