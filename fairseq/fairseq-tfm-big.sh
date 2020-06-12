#!/usr/bin/env bash

#SBATCH --partition=v100
#SBATCH --mem=200G
#SBATCH --time=0-47:59:00
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=20
### SBATCH --gres=gpu:p100:1
#SBATCH --output=R-%x.out.%j
#SBATCH --error=R-%x.err.%j
#SBATCH --export=NONE

# Author : Thamme Gowda
# Date :  May 2020

exp=$1

#exp=exp1

function log {
    printf "$(date): $1\n"
}

function exit_log {
    log "$2"
    exit $1
}

[[ -n $exp ]] || exit_log 1 "ERROR arg1 is required. it should be one of exp{1..11}"

log "experiment = $exp"

source ~/.bashrc

conda deactivate
conda activate fairseq


TEXT=data
SRC=tok.en
TGT=tok.es
BPE_SIZE=32000

TRAIN=$TEXT/$exp/data/train
DEV=$TEXT/tests/dev
TEST=$TEXT/tests/newstest2013
EXP=tfm-big/00-enes-$exp

# == Validate args
for pref in $TRAIN $DEV $TEST; do
    for lang in $SRC $TGT; do
	[[ -f $pref.$lang ]] || exit_log 1 "ERROR: $pref.$lang not found"
    done
done

# ==== BPE 
# At first, BPE was done inside experiment; then I moved to $BPEd dir to reuse b/w base and big models
: <<EOF
DATA_BPE=$EXP/data-bpe
[[ -d $DATA_BPE ]] || mkdir -p $DATA_BPE
BPE_CODES=$DATA_BPE/bpecode$BPE_SIZE.txt
BPE_VOCAB=$DATA_BPE/bpevocab$BPES_ZIE
[[ -f $EXP/_BPEd ]] && log "Skip BPE" || {
    log "Learning Joint BPE on ${TRAIN} $SRC $TGT --> $BPE_CODES"
    subword-nmt learn-joint-bpe-and-vocab -s $BPE_SIZE -i $TRAIN.$SRC $TRAIN.$TGT -o $BPE_CODES --write-vocabulary $BPE_VOCAB.$SRC $BPE_VOCAB.$TGT
    log "BPE learn: done! $(wc -l $BPE_CODES)"

    for lang in $SRC $TGT; do
	printf "train $TRAIN\ndev $DEV\n" | while read name path; do
	    inp=$path.$lang
	    out=$DATA_BPE/$name.$lang.bpe$BPE_SIZE
            log "Apply BPE: $inp -> $out"
	    # as recommended by subword_nmt README, applying the same 
	    # [[ $split == 'train' ]] && thresh="--vocabulary-threshold=5" || thresh=""
	    cmd="subword-nmt apply-bpe -c $BPE_CODES -i $inp -o $out --vocabulary $BPE_VOCAB.$lang"
	    log "RUN: $cmd"
	    eval $cmd || { log "ERROR: BPE failed"; exit 2; }
	done
    done
    touch $EXP/_BPEd
}

EOF

DATA_BPE=$TEXT/$exp/data-bpe
BPE_CODES=$DATA_BPE/bpecode$BPE_SIZE.txt
BPE_VOCAB=$DATA_BPE/bpevocab$BPES_ZIE
[[ -f $BPE_CODES ]] || exit_log 1 "$BPE_CODES file not found"
[[ -f $BPE_VOCAB.$SRC ]] || exit_log 1 "$BPE_VOCAB.$SRC file not found"
[[ -f $BPE_VOCAB.$TGT ]] || exit_log 1 "$BPE_VOCAB.$TGT file not found"


# Check if BPE files exist
SRC_B=$SRC.bpe$BPE_SIZE
TGT_B=$TGT.bpe$BPE_SIZE

for lang in $SRC_B $TGT_B; do
    for pref in $DATA_BPE/train $DATA_BPE/dev ; do
	[[ -f $pref.$lang ]] || exit_log 1 "ERROR: $pref.$lang not found"
    done
done


: <<EOF

# === Prepocess
[[ -f $EXP/_PREPd ]] && log "Skip Preprocess " || {
    log "Proprocessing..."
    cmd="fairseq-preprocess --source-lang $SRC_B --target-lang $TGT_B \
    --trainpref $DATA_BPE/train \
    --validpref $DATA_BPE/dev \
    --destdir $EXP/data-bin  \
    --nwordssrc $BPE_SIZE --nwordstgt $BPE_SIZE \
    --joined-dictionary \
    --bpe subword_nmt \
    --workers 20"

#    --testpref $EXP/data-bpe/test \

    log "RUN: $cmd"
    eval $cmd || { log "ERROR: preprocess failed"; rm -r $EXP/data-bin; exit 3; }

    touch $EXP/_PREPd
}

EOF

# reuse preprocessed 
[[ -d $EXP ]] || mkdir -p $EXP
[[ -d $EXP/data-bin ]] ||  ln -s $(realpath $DATA_BPE/../data-bin )  $EXP/data-bin

# === Training 
#export CUDA_VISIBLE_DEVICES=0
[[ -f $EXP/_TRAINd ]] && log "Skip training" || {
    log "Training..."
 
    cmd="CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train $EXP/data-bin  \
  --task translation --dataset-impl mmap \
  --source-lang $SRC_B --target-lang $TGT_B \
  --bpe subword_nmt \
  --arch transformer_wmt_en_de_big --share-all-embeddings --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr 0.0007 --lr-scheduler inverse_sqrt --warmup-updates 8000 \
  --warmup-init-lr 1e-05 --weight-decay 0.0001 \
  --save-interval-updates=1000 --keep-interval-updates 10 --keep-last-epochs=5 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens 4096 --required-batch-size-multiple 4 --update-freq 2  \
  --save-dir $EXP/models \
  --tensorboard-logdir $EXP/tensorboard \
  --no-progress-bar --log-interval 100 --log-format simple \
  --maximize-best-checkpoint-metric --best-checkpoint-metric bleu \
  --eval-bleu \
  --eval-bleu-args '{\"beam\": 4, \"max_len_a\": 1.2, \"max_len_b\": 10}' \
  --eval-bleu-detok moses \
  --eval-bleu-remove-bpe \
  --max-update=250000 --patience 10
  --fp16" 

#   --eval-bleu-print-samples \
#  --max-update=250000
#  --max-updates=250000 --patience 5  \
# --share-decoder-input-output-embed

    log "RUN: $cmd"
    eval $cmd || exit_log 4 "ERROR: Training failed"
    touch $EXP/_TRAINd
}


[[ -f  $EXP/checkpt.avg5.pt ]] || {
    log "checkpointing "
    FAIRSEQ=/home/07394/tgowda/repos/fairseq
    python $FAIRSEQ/scripts/average_checkpoints.py --inputs $EXP/models --output $EXP/checkpt.avg5.pt --num-update-checkpoints 5 \
	|| exit_log 5 "Error: checkpointing failed"
}



[[ -d $EXP/tests ]] ||  mkdir -p $EXP/tests

test_bpe=$EXP/tests/test.$SRC_B
test_out=$EXP/tests/test.out
test_orig=$TEST.$SRC

[[ -s $test_bpe && "$(wc -l < $test_bpe)" -eq "$(wc -l < $test_orig)" ]] || {

    ln -s $(realpath $test_orig) $EXP/tests/test.$SRC
    log "Apply BPE: $test_orig -> $test_bpe"
    # as recommended by subword_nmt README, applying the same 
    # [[ $split == 'train' ]] && thresh="--vocabulary-threshold=5" || thresh=""
    cmd="subword-nmt apply-bpe -c $BPE_CODES -i $test_orig -o $test_bpe --vocabulary $BPE_VOCAB.$SRC"
    log "RUN: $cmd"
    eval $cmd || exit_log 6 "Error: $cmd"
}


[[ -f  $test_out && "$(wc -l < $test_bpe)" -eq "$(wc -l < $test_out)" ]]  && \
    echo "Skipping decode of $test_bpe -> $test_out" || {
    log "decoding $test_bpe --> $test_out"
    cmd="cat $test_bpe | fairseq-interactive $EXP/data-bin \
 --source-lang $SRC_B --target-lang $TGT_B \
 --path $EXP/checkpt.avg5.pt --lenpen 0.6 --beam 4 \
 | grep '^H' | cut -f3  | sed -r 's/(@@ )|(@@ ?$)//g ' \
  > $test_out"
   log "RUN: $cmd"
   eval "$cmd" || exit_log 8 "ERROR: decode failed"

   cat $test_out | sacremoses -l es detokenize | sacrebleu -l en-es -t wmt13 > $testout.bleu 
}

log "All done"
