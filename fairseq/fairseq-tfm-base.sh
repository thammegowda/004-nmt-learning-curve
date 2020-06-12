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


exp=$1

#exp=exp1

function log {
    printf "$(date): $1\n"
}
[[ -n $exp ]] || {
    log "ERROR arg1 is required. it should be one of exp{1..11}"
    exit 1
}

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
EXP=tfm-base/00-enes-$exp

# == Validate args
for pref in $TRAIN $DEV $TEST; do
    for lang in $SRC $TGT; do
	if [[ ! -f $pref.$lang ]]; then
	    log "ERROR: $pref.$lang not found"
	    exit 1;
	fi
    done
done

# ==== BPE 

[[ -d $EXP/data-bpe ]] || mkdir -p  $EXP/data-bpe

BPE_CODES=$EXP/data-bpe/bpecode$BPE_SIZE.txt
BPE_VOCAB=$EXP/data-bpe/bpevocab$BPES_ZIE

[[ -f $EXP/_BPEd ]] && log "Skip BPE" || {
    log "Learning Joint BPE on ${TRAIN} $SRC $TGT --> $BPE_CODES"
    subword-nmt learn-joint-bpe-and-vocab -s $BPE_SIZE -i $TRAIN.$SRC $TRAIN.$TGT -o $BPE_CODES --write-vocabulary $BPE_VOCAB.$SRC $BPE_VOCAB.$TGT
    log "BPE learn: done! $(wc -l $BPE_CODES)"

    for lang in $SRC $TGT; do
	printf "train $TRAIN\ndev $DEV\n" | while read name path; do
	    inp=$path.$lang
	    out=$EXP/data-bpe/$name.$lang.bpe$BPE_SIZE
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

# === Prepocess
[[ -f $EXP/_PREPd ]] && log "Skip Preprocess " || {
    log "Proprocessing..."
    cmd="fairseq-preprocess --source-lang $SRC.bpe$BPE_SIZE --target-lang $TGT.bpe$BPE_SIZE \
    --trainpref $EXP/data-bpe/train \
    --validpref $EXP/data-bpe/dev \
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


# === Training 
#export CUDA_VISIBLE_DEVICES=0
[[ -f $EXP/_TRAINd ]] && log "Skip training" || {
    log "Training..."
    cmd="CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train $EXP/data-bin  \
  --task translation --dataset-impl mmap \
  --source-lang $SRC.bpe$BPE_SIZE --target-lang $TGT.bpe$BPE_SIZE \
  --bpe subword_nmt \
  --arch transformer --share-all-embeddings --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --warmup-init-lr 1e-07 --weight-decay 0.0001 \
  --max-update=250000 --save-interval-updates=1000 --keep-interval-updates 6 --keep-last-epochs=6 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens 6250 --required-batch-size-multiple 4 \
  --save-dir $EXP/models \
  --tensorboard-logdir $EXP/tensorboard \
  --no-progress-bar --log-interval 100 --log-format simple \
  --eval-bleu \
  --eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' \
  --eval-bleu-detok moses \
  --eval-bleu-remove-bpe \
  --eval-bleu-print-samples \
  --maximize-best-checkpoint-metric \
  --fp16" 

#  --max-updates=250000 --patience 5  \
# --share-decoder-input-output-embed

# These are only on master as of now; TODO: build from master branch
#  --eval-bleu \
#  --eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' \
#  --eval-bleu-detok moses \
#  --eval-bleu-remove-bpe \
#  --eval-bleu-print-samples \

    log "RUN: $cmd"
    eval $cmd || { log "ERROR: Training failed"; exit 4; }
    touch $EXP/_TRAINd
}


[[ -f  $EXP/checkpt.avg5.pt ]] || {
    log "checkpointing "
    FAIRSEQ=/home/07394/tgowda/repos/fairseq
    python $FAIRSEQ/scripts/average_checkpoints.py --inputs $EXP/models --output $EXP/checkpt.avg5.pt --num-update-checkpoints 5 \
	|| { log "Error: checkpointing failed"; exit 5; }
}


log "checkpointing "
[[ -d $EXP/tests ]] ||  mkdir -p $EXP/tests

test_bpe=$EXP/tests/test.bpe
test_out=$EXP/tests/test.out
test_orig=$TEST.$SRC

[[ -s $test_bpe && "$(wc -l < $test_bpe)" -eq "$(wc -l < $test_orig)" ]] || {

    ln -s $(realpath $test_orig) $EXP/tests/test.$SRC
    log "Apply BPE: $test_orig -> $test_bpe"
    # as recommended by subword_nmt README, applying the same 
    # [[ $split == 'train' ]] && thresh="--vocabulary-threshold=5" || thresh=""
    cmd="subword-nmt apply-bpe -c $BPE_CODES -i $test_orig -o $test_bpe --vocabulary $BPE_VOCAB.$SRC"
    log "RUN: $cmd"
    eval $cmd || {
	log "Error: $cmd"
	exit 6
    }
}


[[ -f  $test_out && "$(wc -l < $test_bpe)" -eq "$(wc -l < $test_out)" ]] \
  && echo "Skipping decode of $test_bpe -> $test_out" \
  || {
    log "decoding $test_bpe --> $test_out"
    cmd="cat $test_bpe | fairseq-interactive $EXP/data-bin \
 --source-lang $SRC.bpe$BPE_SIZE --target-lang $TGT.bpe$BPE_SIZE \
 --path $EXP/checkpt.avg5.pt --lenpen 0.6 --beam 4 \
 | grep '^H' | cut -f3  | sed -r 's/(@@ )|(@@ ?$)//g' \
   > $test_out"
   log "RUN: $cmd"
   eval "$cmd" || { log "ERROR: decode failed"; exit 8; }
}

log "All done"
