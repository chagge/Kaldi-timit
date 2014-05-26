#!/bin/bash

# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script trains a DNN on top of fMLLR features. 
# The training is done in 3 stages,
#
# 1) RBM pre-training:
#    in this unsupervised stage we train stack of RBMs, 
#    a good starting point for frame cross-entropy trainig.
# 2) frame cross-entropy training:
#    the objective is to classify frames to correct pdfs.
# 3) sequence-training optimizing sMBR: 
#    the objective is to emphasize state-sequences with better 
#    frame accuracy w.r.t. reference alignment.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

njobs=6

# Config:
gmmdir=exp/tri3

### Align dev data
{
  # steps/align_fmllr.sh --nj $njobs --cmd "$train_cmd" \
  #  data/train data/lang exp/tri3 exp/tri3_ali

  steps/align_fmllr.sh --nj $njobs --cmd "$train_cmd" \
    data/dev data/lang $gmmdir ${gmmdir}_dev_ali || exit 1;
}

data_fmllr=data_fmllr_tri3b
stage=0 # resume training with --stage=N



if [ $stage -le 0 ]; then
  # Store fMLLR features, so we can train on them easily,
  # dev
  dir=$data_fmllr/dev
  steps/nnet/make_fmllr_feats.sh --nj $njobs --cmd "$train_cmd" \
     --transform-dir $gmmdir/decode_dev \
     $dir data/dev $gmmdir $dir/log $dir/data || exit 1
  # test
  dir=$data_fmllr/test
  steps/nnet/make_fmllr_feats.sh --nj $njobs --cmd "$train_cmd" \
     --transform-dir $gmmdir/decode_test \
     $dir data/test $gmmdir $dir/log $dir/data || exit 1
  # train
  dir=$data_fmllr/train
  steps/nnet/make_fmllr_feats.sh --nj $njobs --cmd "$train_cmd" \
     --transform-dir ${gmmdir}_ali \
     $dir data/train $gmmdir $dir/log $dir/data || exit 1
fi

if [ $stage -le 1 ]; then
  # Pre-train DBN, i.e. a stack of RBMs
  dir=exp/dnn4b_pretrain-dbn
  (tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh --rbm-iter 3 $data_fmllr/train $dir || exit 1;
fi

if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  dir=exp/dnn4b_pretrain-dbn_dnn
  train_ali=${gmmdir}_ali
  dev_ali=${gmmdir}_dev_ali
  feature_transform=exp/dnn4b_pretrain-dbn/final.feature_transform
  dbn=exp/dnn4b_pretrain-dbn/6.dbn
  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    $data_fmllr/train $data_fmllr/dev data/lang $train_ali $dev_ali $dir || exit 1;
  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj $njobs --use-gpu "yes" --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmmdir/graph $data_fmllr/dev $dir/decode_dev || exit 1;
  steps/nnet/decode.sh --nj $njobs --use-gpu "yes" --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    $gmmdir/graph $data_fmllr/test $dir/decode_test || exit 1;
fi
# ###

# Sequence training using sMBR criterion, we do Stochastic-GD 
# with per-utterance updates. We use usually good acwt 0.1
# Lattices are re-generated after 1st epoch, to get faster convergence.
dir=exp/dnn4b_pretrain-dbn_dnn_smbr
srcdir=exp/dnn4b_pretrain-dbn_dnn
acwt=0.1

if [ $stage -le 3 ]; then
  # First we generate lattices and alignments:
  steps/nnet/align.sh --nj $njobs --use-gpu "yes" --cmd "$train_cmd" \
    $data_fmllr/train data/lang $srcdir ${srcdir}_ali || exit 1;
  steps/nnet/make_denlats.sh --nj $njobs --use-gpu "yes" --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
    $data_fmllr/train data/lang $srcdir ${srcdir}_denlats || exit 1;
fi

if [ $stage -le 4 ]; then
  # Re-train the DNN by 1 iteration of sMBR 
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 1 --acwt $acwt --do-smbr true \
    $data_fmllr/train data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
  # Decode (reuse HCLG graph)
  for ITER in 1; do
    steps/nnet/decode.sh --nj $njobs --use-gpu "yes" --cmd "$decode_cmd" --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $gmmdir/graph $data_fmllr/dev $dir/decode_dev_it${ITER} || exit 1;
    steps/nnet/decode.sh --nj $njobs --use-gpu "yes" --cmd "$decode_cmd" --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $gmmdir/graph $data_fmllr/test $dir/decode_test_it${ITER} || exit 1;
  done 
fi

# Re-generate lattices, run 4 more sMBR iterations
dir=exp/dnn4b_pretrain-dbn_dnn_smbr_i1lats
srcdir=exp/dnn4b_pretrain-dbn_dnn_smbr
acwt=0.1

if [ $stage -le 5 ]; then
  # Generate lattices and alignments:
  steps/nnet/align.sh --nj $njobs --use-gpu "yes" --cmd "$train_cmd" \
    $data_fmllr/train data/lang $srcdir ${srcdir}_ali || exit 1;
  steps/nnet/make_denlats.sh --nj $njobs --use-gpu "yes" --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
    $data_fmllr/train data/lang $srcdir ${srcdir}_denlats || exit 1;
fi

if [ $stage -le 6 ]; then
  # Re-train the DNN by 1 iteration of sMBR 
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 4 --acwt $acwt --do-smbr true \
    $data_fmllr/train data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
  # Decode (reuse HCLG graph)
  for ITER in 1 2 3 4; do
    steps/nnet/decode.sh --nj $njobs --use-gpu "yes" --cmd "$decode_cmd" --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $gmmdir/graph $data_fmllr/dev $dir/decode_dev_iter${ITER} || exit 1;
    steps/nnet/decode.sh --nj $njobs --use-gpu "yes" --cmd "$decode_cmd" --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $gmmdir/graph $data_fmllr/test $dir/decode_test_iter${ITER} || exit 1;
  done 
fi

#Getting results [see RESULTS file]
for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
