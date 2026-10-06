# Train
conda run -n closure python /home/me/disk-256/00_RND_MODELS/paddlepaddle/siglip2/train_siglip2_attributes.py \
  --export-dir /home/me/disk-256/00_RND_MODELS/paddlepaddle/siglip2/gender-age-1 \
  --config /home/me/disk-256/00_RND_MODELS/paddlepaddle/dataset-collector/server/template/siglip2/siglip2_gender_age/config.yaml \
  --output-dir "$HOME/siglip2-runs/gender-age-1" \
  --checkpoint google/siglip2-base-patch16-224 \
  --epochs 15 --batch-size 32 --balanced-loss


# Quantize
  conda run -n closure python /home/me/disk-256/00_RND_MODELS/paddlepaddle/siglip2/quantize_siglip2.py \
  --checkpoint $HOME/siglip2-runs/gender-age-2/best.pt \
  --export-dir /home/me/disk-256/00_RND_MODELS/paddlepaddle/siglip2/gender-age-1 \
  --config /home/me/disk-256/00_RND_MODELS/paddlepaddle/dataset-collector/server/template/siglip2/siglip2_gender_age/config.yaml \
  --output-dir $HOME/siglip2-runs/gender-age-2-quant \
  --mode fp16