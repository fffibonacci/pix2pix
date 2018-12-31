#400 b2a
CUDA_VISIBLE_DEVICES=1\
python ./scripts/eval_cityscapes/evaluate.py \
--cityscapes_dir ~/Github/gtFine_val \
--result_dir ~/Github/Self-inverse/results/cityscapes_pix2pix_Self_allLrelu400epoch_v4/test_latest/images/\
--output_dir ~/Github/pix2pix/outputs/cityscapes_pix2pix_Self_allLrelu400AtoB
