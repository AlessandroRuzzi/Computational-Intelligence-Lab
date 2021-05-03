# Train on 64 GB of RAM, 1 GPU (Tesla with 32GB memory)
# bsub -R "rusage[mem=64000, ngpus_excl_p=1]" -R "select[gpu_model0==TeslaV100_SXM2_32GB]" python3 ./run.py trainer.gpus=1
# Train on 64 GB of RAM, 1 CPU 
# bsub -R "rusage[mem=64000]" python3 ./run.py 
# Train on 64 GB of RAM, 8 GPUs (GeForceGTX1080 with 8GB of memory)
#bsub -W 20:00 -R "rusage[mem=64000, ngpus_excl_p=1]" python3 ./run.py +experiment=exp_standard
bsub -W 24:00 -R "rusage[mem=64000, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" python3 ./run.py +experiment=exp_simple_unet_backboned_fine_tuning
# Run test from checkpoint 
bsub -W 01:00 -R "rusage[mem=64000, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" python3 ./run.py +experiment=exp__008 +model.checkpoint_path=/cluster/home/scflavio/Computational-Intelligence-Lab-2021/logs/ckpts/2021-04-29/15-18-43/'epoch=148-val_kaggle=0.96.ckpt'

# Run test f008 with checkpoint 
bsub -W 01:00 -R "rusage[mem=64000, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" python3 ./run.py +experiment=test__f008 +model.checkpoint_path=/cluster/home/scflavio/Computational-Intelligence-
Lab-2021/logs/ckpts/2021-04-29/15-18-43/'epoch\=148-val_kaggle\=0.96.ckpt'