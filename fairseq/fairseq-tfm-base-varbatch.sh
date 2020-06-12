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
EXP=tfm-base-varbatch/00-enes-$exp

#exp=exp1

function log {
    printf "$(date --rfc-3339 s): $1\n"
}

function exit_log {
    log "$2"
    exit $1
}

log "experiment = $exp"
[[ -n $exp ]] || exit_log 1 "ERROR arg1 is required. it should be one of exp{1..11}"

# [exp#] = BPE_SIZE N_GPU UPD_FREQ TOKS_PER_GPU
declare -A BATCHTAB=(
    [exp1]="6000 4 1 512" 
    [exp2]="6000 4 1 512"
    [exp3]="6000 4 1 512"
    [exp4]="8000 4 1 1024"
    [exp5]="8000 4 1 2048"
    [exp6]="16000 4 1 4096"
    [exp7]="16000 4 1 6250"
    [exp8]="32000 4 2 6250"
    [exp9]="32000 4 4 6250"
    [exp10]="64000 4 16 3125"
    [exp11]="64000 4 32 3125" ) 

# 64,000 vocabulary takes more memory, so per GPU toks are reduced


[[ ${BATCHTAB[$exp]+Y} ]] || exit_log 1 "cant find settings for $exp in ${BATCHTAB}"
log "$exp : ${BATCHTAB[$exp]}"
BPE_SIZE=$(echo "${BATCHTAB[$exp]}" | cut -d' ' -f 1)
N_GPU=$(echo "${BATCHTAB[$exp]}" | cut -d' ' -f 2)
UPD_FREQ=$(echo "${BATCHTAB[$exp]}" | cut -d' ' -f 3)
TOKS_PER_GPU=$(echo "${BATCHTAB[$exp]}" | cut -d' ' -f 4)
[[ -n $BPE_SIZE && -n $N_GPU && -n $UPD_FREQ && $TOKS_PER_GPU ]] || exit_log 1 "Couldnt parse the bpe/batch args"

# 64,000 vocabulary takes more memory, so per GPU toks are reduced
[[ $BPE_SIZE -le "32000" ]] && VALID_BATCH_TOKS=25000 || VALID_BATCH_TOKS=12000

log "$exp : BPE_SIZE=$BPE_SIZE"
log "$exp : N_GPU*UPD_FREQ*TOKS_PER_GPU=$N_GPU*$UPD_FREQ*$TOKS_PER_GPU = $(($N_GPU*$UPD_FREQ*$TOKS_PER_GPU))"

source ~/.bashrc

conda deactivate
conda activate fairseq


TEXT=data
SRC=tok.en
TGT=tok.es
#BPE_SIZE=32000

TRAIN=$TEXT/$exp/data/train
DEV=$TEXT/tests/dev
TEST=$TEXT/tests/newstest2013


# == Validate args
for pref in $TRAIN $DEV $TEST; do
    for lang in $SRC $TGT; do
	[[ -f $pref.$lang ]] || exit_log 1 "ERROR: $pref.$lang not found"
    done
done

# ==== BPE 
#DATA_BPE=$EXP/data-bpe
DATA_BPE=$TEXT/$exp/data-bpe-$BPE_SIZE
DATA_BIN=$TEXT/$exp/data-bin-$BPE_SIZE
BPE_CODES=$DATA_BPE/bpecode$BPE_SIZE.txt
BPE_VOCAB=$DATA_BPE/bpevocab$BPES_ZIE
SRC_B=$SRC.bpe$BPE_SIZE
TGT_B=$TGT.bpe$BPE_SIZE


[[ -d $DATA_BPE ]] || mkdir -p $DATA_BPE
[[ -d $DATA_BIN ]] || mkdir -p $DATA_BIN

[[ -f $DATA_BPE/_BPEd ]] && log "Skip BPE" || {
    log "Learning Joint BPE on ${TRAIN} $SRC $TGT --> $BPE_CODES"
    subword-nmt learn-joint-bpe-and-vocab -s $BPE_SIZE -i $TRAIN.$SRC $TRAIN.$TGT -o $BPE_CODES \
	--write-vocabulary $BPE_VOCAB.$SRC $BPE_VOCAB.$TGT
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
    touch $DATA_BPE/_BPEd
}


#DATA_BPE=$TEXT/$exp/data-bpe-$BPE_SIZE
[[ -f $BPE_CODES ]] || exit_log 1 "$BPE_CODES file not found"
[[ -f $BPE_VOCAB.$SRC ]] || exit_log 1 "$BPE_VOCAB.$SRC file not found"
[[ -f $BPE_VOCAB.$TGT ]] || exit_log 1 "$BPE_VOCAB.$TGT file not found"

# Check if BPE files exist
for lang in $SRC_B $TGT_B; do
    for pref in $DATA_BPE/train $DATA_BPE/dev ; do
	[[ -f $pref.$lang ]] || exit_log 1 "ERROR: $pref.$lang not found"
    done
done

# === Prepocess
[[ -f $DATA_BIN/_PREPd ]] && log "Skip Preprocess " || {
    log "Proprocessing $DATA_BPE -> $DATA_BIN..."
    cmd="fairseq-preprocess --source-lang $SRC_B --target-lang $TGT_B \
    --trainpref $DATA_BPE/train \
    --validpref $DATA_BPE/dev \
    --destdir $DATA_BIN  \
    --nwordssrc $BPE_SIZE --nwordstgt $BPE_SIZE \
    --joined-dictionary \
    --bpe subword_nmt \
    --workers 20"

#    --testpref $EXP/data-bpe/test \
    log "RUN: $cmd"
    eval $cmd || { log "ERROR: preprocess failed"; rm -r $EXP/data-bin; exit 3; }
    touch $DATA_BIN/_PREPd
}

# reuse 
[[ -d $EXP ]] || mkdir -p $EXP

# === Training 
#export CUDA_VISIBLE_DEVICES=0
[[ -f $EXP/_TRAINd ]] && log "Skip training" || {
    log "Training..."
    DEVICES=$(python -c "print(','.join(map(str, range(0, $N_GPU))))")
    cmd="CUDA_VISIBLE_DEVICES=$DEVICES fairseq-train $DATA_BIN  \
  --task translation --dataset-impl mmap \
  --source-lang $SRC_B --target-lang $TGT_B \
  --bpe subword_nmt \
  --arch transformer --share-all-embeddings --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr 0.0007 --lr-scheduler inverse_sqrt --warmup-updates 8000 \
  --warmup-init-lr 1e-07 --weight-decay 0.0001 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens $TOKS_PER_GPU --required-batch-size-multiple $N_GPU --update-freq $UPD_FREQ  \
  --max-tokens-valid $VALID_BATCH_TOKS \
  --save-dir $EXP/models \
  --tensorboard-logdir $EXP/tensorboard \
  --no-progress-bar --log-interval 50 --log-format simple \
  --maximize-best-checkpoint-metric \
  --best-checkpoint-metric bleu \
  --eval-bleu \
  --eval-bleu-args '{\"beam\": 4, \"max_len_a\": 1.2, \"max_len_b\": 10}' \
  --eval-bleu-detok moses \
  --eval-bleu-remove-bpe \
  --save-interval-updates=1000 --keep-interval-updates 20 --validate-interval 1000 \
  --max-update=400000 --patience 20 \
  --fp16" 


#  --best-checkpoint-metric bleu \
#  --eval-bleu-print-samples \
#  --eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' \
# 
# --max-updates=250000 --patience 5  \
# --share-decoder-input-output-embed

    log "RUN: $cmd"
    eval $cmd || exit_log 4 "ERROR: Training failed"
    touch $EXP/_TRAINd
}


[[ -f  $EXP/checkpt.avg5.pt ]] || {
    log "checkpointing "
    FAIRSEQ=$HOME/repos/fairseq
    python $FAIRSEQ/scripts/average_checkpoints.py --inputs $EXP/models --output $EXP/checkpt.avg5.pt --num-update-checkpoints 5 \
	|| exit_log 5 "Error: checkpointing failed"
}


log "checkpointing "
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
    cmd="cat $test_bpe | fairseq-interactive $DATA_BIN \
 --source-lang $SRC_B --target-lang $TGT_B \
 --path $EXP/checkpt.avg5.pt --lenpen 0.6 --beam 4 \
 | grep '^H' | cut -f3  | sed -r 's/(@@ )|(@@ ?$)//g' \
  > $test_out"
   log "RUN: $cmd"
   eval "$cmd" || exit_log 8 "ERROR: decode failed"
}

log "All done"
