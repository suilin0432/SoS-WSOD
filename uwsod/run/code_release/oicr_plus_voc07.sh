CUDA_VISIBLE_DEVICES=0,1,2,3 python3 projects/WSL/tools/train_net_multi.py --num-gpus 4 \
	--config-file projects/WSL/configs/Detection/code_release/voc07_oicr_plus.yaml \
	--dist-url tcp://0.0.0.0:8723 \
	OUTPUT_DIR output/sos_release_oicr_plus/
