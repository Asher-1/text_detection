GPU=0
CUDA_VISIBLE_DEVICES=${GPU} \
python main.py \
	--train_nips "/media/yons/data/dataset/images/text_data/MORAN/IIIT5K_3000" \
	--train_cvpr "/media/yons/data/dataset/images/text_data/MORAN/ic15_2077" \
	--valroot "/media/yons/data/dataset/images/text_data/MORAN/ic13_1015" \
	--workers 2 \
	--batchSize 64 \
	--niter 10 \
	--lr 0.1 \
	--cuda \
	--experiment output/ \
	--displayInterval 100 \
	--valInterval 1000 \
	--saveInterval 40000 \
	--adadelta \
	--BidirDecoder