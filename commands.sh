#!/bin/bash

# 3 different model structure settings: simplified_bpr_text, simplified_bpr_image, simplified_bpr_text_image

## data prepare
# movies:
python ./scripts/index_and_filter_review_file.py \
    /hdd3/haotao/amazon_review_dataset/reviews_Movies_and_TV_5.json.gz \
    /hdd3/haotao/amazon_review_dataset/reviews_Movies_and_TV_5/ \
    1

python ./scripts/split_train_test.py \
    /hdd3/haotao/amazon_review_dataset/reviews_Movies_and_TV_5/min_count1/ \
    0.3

python ./scripts/match_with_image_features.py \
    /hdd3/haotao/amazon_review_dataset/reviews_Movies_and_TV_5/min_count1/ \
    /hdd3/haotao/amazon_review_dataset/image_features_Movies_and_TV.b

# CD:
python ./scripts/index_and_filter_review_file.py \
    /hdd3/haotao/amazon_review_dataset/reviews_CDs_and_Vinyl_5.json.gz \
    /hdd3/haotao/amazon_review_dataset/reviews_CDs_and_Vinyl_5/ \
    1

python ./scripts/split_train_test.py \
    /hdd3/haotao/amazon_review_dataset/reviews_CDs_and_Vinyl_5/min_count1/ \
    0.3

python ./scripts/match_with_image_features.py \
    /hdd3/haotao/amazon_review_dataset/reviews_CDs_and_Vinyl_5/min_count1/ \
    /hdd3/haotao/amazon_review_dataset/image_features_CDs_and_Vinyl.b

# clothing:
python ./scripts/index_and_filter_review_file.py \
    /hdd3/haotao/amazon_review_dataset/reviews_Clothing_Shoes_and_Jewelry_5.json.gz \
    /hdd3/haotao/amazon_review_dataset/reviews_Clothing_Shoes_and_Jewelry_5/ \
    1

python ./scripts/split_train_test.py \
    /hdd3/haotao/amazon_review_dataset/reviews_Clothing_Shoes_and_Jewelry_5/min_count1/ \
    0.3

python ./scripts/match_with_image_features.py \
    /hdd3/haotao/amazon_review_dataset/reviews_Clothing_Shoes_and_Jewelry_5/min_count1/ \
    /hdd3/haotao/amazon_review_dataset/image_features_Clothing_Shoes_and_Jewelry.b

# beauty:
python ./scripts/index_and_filter_review_file.py \
    /hdd3/haotao/amazon_review_dataset/reviews_Beauty_5.json.gz \
    /hdd3/haotao/amazon_review_dataset/reviews_Beauty_5/ \
    1

python ./scripts/split_train_test.py \
    /hdd3/haotao/amazon_review_dataset/reviews_Beauty_5/min_count1/ \
    0.3

python ./scripts/match_with_image_features.py \
    /hdd3/haotao/amazon_review_dataset/reviews_Beauty_5/min_count1/ \
    /hdd3/haotao/amazon_review_dataset/image_features_Beauty.b

# cell phones:
python ./scripts/index_and_filter_review_file.py \
    /hdd3/haotao/amazon_review_dataset/reviews_Cell_Phones_and_Accessories_5.json.gz \
    /hdd3/haotao/amazon_review_dataset/reviews_Cell_Phones_and_Accessories_5/ \
    1

python ./scripts/split_train_test.py \
    /hdd3/haotao/amazon_review_dataset/reviews_Cell_Phones_and_Accessories_5/min_count1/ \
    0.3

python ./scripts/match_with_image_features.py \
    /hdd3/haotao/amazon_review_dataset/reviews_Cell_Phones_and_Accessories_5/min_count1/ \
    /hdd3/haotao/amazon_review_dataset/image_features_Cell_Phones_and_Accessories.b

## train:

# movies:
CUDA_VISIBLE_DEVICES=0 python ./JRL/main.py \
    --embed_size 300 \
    --data_dir /hdd3/haotao/amazon_review_dataset/reviews_Movies_and_TV_5/min_count1/ \
    --input_train_dir /hdd3/haotao/amazon_review_dataset/reviews_Movies_and_TV_5/min_count1/ \
    --net_struct simplified_bpr_text

CUDA_VISIBLE_DEVICES=2 python ./JRL/main.py \
    --embed_size 300 \
    --data_dir /hdd3/haotao/amazon_review_dataset/reviews_Movies_and_TV_5/min_count1/ \
    --input_train_dir /hdd3/haotao/amazon_review_dataset/reviews_Movies_and_TV_5/min_count1/ \
    --net_struct simplified_bpr_text_image

# CDs:
CUDA_VISIBLE_DEVICES=0 python ./JRL/main.py \
    --embed_size 300 \
    --data_dir /hdd3/haotao/amazon_review_dataset/reviews_Movies_and_TV_5/min_count1/ \
    --input_train_dir /hdd3/haotao/amazon_review_dataset/reviews_Movies_and_TV_5/min_count1/ \
    --net_struct simplified_bpr_text

CUDA_VISIBLE_DEVICES=2 python ./JRL/main.py \
    --embed_size 300 \
    --data_dir /hdd3/haotao/amazon_review_dataset/reviews_Movies_and_TV_5/min_count1/ \
    --input_train_dir /hdd3/haotao/amazon_review_dataset/reviews_Movies_and_TV_5/min_count1/ \
    --net_struct simplified_bpr_text_image

# clothing:
CUDA_VISIBLE_DEVICES=6 python ./JRL/main.py \
    --embed_size 300 \
    --data_dir /hdd3/haotao/amazon_review_dataset/reviews_Clothing_Shoes_and_Jewelry_5/min_count1/ \
    --input_train_dir /hdd3/haotao/amazon_review_dataset/reviews_Clothing_Shoes_and_Jewelry_5/min_count1/ \
    --net_struct simplified_bpr_text

CUDA_VISIBLE_DEVICES=1 python ./JRL/main.py \
    --embed_size 300 \
    --image_weight 0.001 \
    --data_dir /hdd3/haotao/amazon_review_dataset/reviews_Clothing_Shoes_and_Jewelry_5/min_count1/ \
    --input_train_dir /hdd3/haotao/amazon_review_dataset/reviews_Clothing_Shoes_and_Jewelry_5/min_count1/ \
    --net_struct simplified_bpr_text_image

# beauty:
CUDA_VISIBLE_DEVICES=0 python ./JRL/main.py \
    --embed_size 300 \
    --data_dir /hdd3/haotao/amazon_review_dataset/reviews_Beauty_5/min_count1/ \
    --input_train_dir /hdd3/haotao/amazon_review_dataset/reviews_Beauty_5/min_count1/ \
    --net_struct simplified_bpr_text

CUDA_VISIBLE_DEVICES=2 python ./JRL/main.py \
    --embed_size 300 \
    --data_dir /hdd3/haotao/amazon_review_dataset/reviews_Beauty_5/min_count1/ \
    --input_train_dir /hdd3/haotao/amazon_review_dataset/reviews_Beauty_5/min_count1/ \
    --net_struct simplified_bpr_text_image

CUDA_VISIBLE_DEVICES=0 python ./JRL/main.py \
    --embed_size 150 \
    --data_dir /hdd3/haotao/amazon_review_dataset/reviews_Beauty_5/min_count1/ \
    --input_train_dir /hdd3/haotao/amazon_review_dataset/reviews_Beauty_5/min_count1/ \
    --net_struct simplified_bpr_text_image

# cell phones:
CUDA_VISIBLE_DEVICES=1 python ./JRL/main.py \
    --embed_size 300 \
    --data_dir /hdd3/haotao/amazon_review_dataset/reviews_Cell_Phones_and_Accessories_5/min_count1/ \
    --input_train_dir /hdd3/haotao/amazon_review_dataset/reviews_Cell_Phones_and_Accessories_5/min_count1/ \
    --net_struct simplified_bpr_text

CUDA_VISIBLE_DEVICES=4 python ./JRL/main.py \
    --embed_size 300 \
    --data_dir /hdd3/haotao/amazon_review_dataset/reviews_Cell_Phones_and_Accessories_5/min_count1/ \
    --input_train_dir /hdd3/haotao/amazon_review_dataset/reviews_Cell_Phones_and_Accessories_5/min_count1/ \
    --net_struct simplified_bpr_text_image

# eval:
CUDA_VISIBLE_DEVICES=6 python ./JRL/main.py \
    --embed_size 300 \
    --steps_per_checkpoint 1 \
    --data_dir /hdd3/haotao/amazon_review_dataset/reviews_Clothing_Shoes_and_Jewelry_5/min_count1/ \
    --input_train_dir /hdd3/haotao/amazon_review_dataset/reviews_Clothing_Shoes_and_Jewelry_5/min_count1/ \
    --train_dir ./results/Clothing_Shoes_and_Jewelry_5/emb300/ \
    --net_struct simplified_bpr_text \
    --decode True

python scripts/recommendation_metric.py \
    ./results/Clothing_Shoes_and_Jewelry_5/emb300/test.product.ranklist \
    /hdd3/haotao/amazon_review_dataset/reviews_Clothing_Shoes_and_Jewelry_5/min_count1/test.qrels \
    10

CUDA_VISIBLE_DEVICES=5 python ./JRL/main.py \
    --embed_size 300 \
    --steps_per_checkpoint 1 \
    --data_dir /hdd3/haotao/amazon_review_dataset/reviews_Beauty_5/min_count1/ \
    --input_train_dir /hdd3/haotao/amazon_review_dataset/reviews_Beauty_5/min_count1/ \
    --net_struct simplified_bpr_text \
    --decode True

python scripts/recommendation_metric.py \
    ./results/reviews_Beauty_5/simplified_bpr_text/emb300/test.product.ranklist \
    /hdd3/haotao/amazon_review_dataset/reviews_Beauty_5/min_count1/test.qrels \
    10
