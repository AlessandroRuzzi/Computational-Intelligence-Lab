# Train on 64 GB of RAM, 1 GPU (Tesla with 32GB memory)
# bsub -R "rusage[mem=64000, ngpus_excl_p=1]" -R "select[gpu_model0==TeslaV100_SXM2_32GB]" python3 ./run.py trainer.gpus=1
# Train on 64 GB of RAM, 1 CPU 
# bsub -R "rusage[mem=64000]" python3 ./run.py 
# Train on 64 GB of RAM, 8 GPUs (GeForceGTX1080 with 8GB of memory)
bsub -R "rusage[mem=64000, ngpus_excl_p=1]" python3 ./run.py trainer.gpus=1 datamodule.num_workers=0

# bsub -R "rusage[mem=64000, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" python3 ./run.py trainer.gpus=1

