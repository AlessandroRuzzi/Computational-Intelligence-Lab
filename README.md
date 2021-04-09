# Computational-Intelligence-Lab-2021
Repository for the Computational Intelligence Lab course project at ETH Zurich


## How to run on Leonhard Cluster 

0. Clone repository into cluster. 

 
1. Copy new .env file and modify it by adding your environment variables:
```console
cp .env.tmp .env 
vim .env 
```

2. Make ``leonhard_init.sh`` executable:
```console
chmod +x leonhard_init.sh
```

3. Create virtual environment called ``venv``: 
```console
python -m venv venv
```

4. Run ``leonhard_init.sh`` to install modules, activate virtual environment, and install packages from ``requirements.txt``: 
```console
./leonhard_init.sh
``` 

5. Run model on a single GPU:
```console
bsub -R "rusage[ngpus_excl_p=1]" ./run.py trainer.gpus=1
``` 
