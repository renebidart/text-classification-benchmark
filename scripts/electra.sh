#!/bin/bash
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=45G
#SBATCH --time=1-0:00:00
#SBATCH --account=def-mjshafie
#SBATCH --output slurm-%x-%j.out

module load python/3.8
source ~/ENV/bin/activate
module load StdEnv/2020
module load gcc/9.3.0
module load cuda/11.4
module load arrow/8.0.0
pip install datasets==2.2.2
pip install transformers==4.19.2

MODEL="google/electra-base-discriminator"

TASK_NAMES='sst2 qqp mnli qnli boolq ag_news imdb snli rotten_tomatoes yelp_polarity eraser_multi_rc wiki_qa scitail emotion tweet_eval_hate'

for TASK_NAME in $TASK_NAMES
do
    CUDA_VISIBLE_DEVICES=0 python run_classifier.py \
    --model_name_or_path $MODEL \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --metric_for_best_model eval_loss \
    --load_best_model_at_end \
    --max_seq_length 512 \
    --fp16 \
    --report_to none \
    --output_dir output/${MODEL}/${TASK_NAME} \
    --overwrite_output_dir
done