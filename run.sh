# Train
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_net.py --cfg configs/VGG-Sound/SLOWFAST_R50.yaml NUM_GPUS 4 OUTPUT_DIR /data/blai38/Models/VGGSound/slowfast_r50 VGGSOUND.AUDIO_DATA_DIR /data/blai38/Datasets/VGGSound_final/audio_16kHz VGGSOUND.ANNOTATIONS_DIR /data/blai38/Datasets/VGGSound_final
python tools/run_net.py --cfg configs/VGG-Sound/SLOWFAST_R50.yaml NUM_GPUS 4 OUTPUT_DIR /srv/rehg-lab/flash6/blai38/Models/VGGSound/slowfast_r50 VGGSOUND.AUDIO_DATA_DIR /srv/rehg-lab/flash6/blai38/Datasets/VGGSound_final/audio_16kHz VGGSOUND.ANNOTATIONS_DIR /srv/rehg-lab/flash6/blai38/Datasets/VGGSound_final

# Test
CUDA_VISIBLE_DEVICES=0 python tools/run_net.py --cfg configs/VGG-Sound/SLOWFAST_R50.yaml NUM_GPUS 1 OUTPUT_DIR /data/blai38/Models/VGGSound/slowfast_r50 VGGSOUND.AUDIO_DATA_DIR /data/blai38/Datasets/VGGSound_final/audio_16kHz VGGSOUND.ANNOTATIONS_DIR /data/blai38/Datasets/VGGSound_final TRAIN.ENABLE False TEST.ENABLE True TEST.CHECKPOINT_FILE_PATH /data/blai38/Models/pretrained/vggsound/SLOWFAST_VGG.pyth
python tools/run_net.py --cfg configs/VGG-Sound/SLOWFAST_R50.yaml NUM_GPUS 1 OUTPUT_DIR /srv/rehg-lab/flash6/blai38/Models/VGGSound/slowfast_r50 VGGSOUND.AUDIO_DATA_DIR /srv/rehg-lab/flash6/blai38/Datasets/VGGSound_final/audio_16kHz VGGSOUND.ANNOTATIONS_DIR /srv/rehg-lab/flash6/blai38/Datasets/VGGSound_final TRAIN.ENABLE False TEST.ENABLE True TEST.CHECKPOINT_FILE_PATH /srv/rehg-lab/flash6/blai38/Models/pretrained/vggsound/SLOWFAST_VGG.pyth

# SkyNet prefix
srun --constraint=2080_ti --gpus-per-node=1 -c 7 --pty bash
srun -p short --constraint=2080_ti --gpus-per-node=4 -c 28 -J Sound
srun -p short --constraint=2080_ti --gpus-per-node=1 -c 10 -J Sound

