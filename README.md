# Computational-Intelligence-Lab-2021
Repository for the Computational Intelligence Lab course project at ETH Zurich

## How to reproduce results

**Step 1:** Clone the repository:
```console
git clone https://github.com/AlessandroRuzzi/Computational-Intelligence-Lab-2021
```

**Step 2:** Copy new .env file and modify it by adding your environment variables,:
```console
cp .env.tmp .env 
vim .env 
```

Example of ``.env`` file:

```console
COMET_API_KEY=Your Key
COMET_WORKSPACE=alessandroruzzi
KAGGLE_USERNAME=alessandroruzzi
KAGGLE_KEY=Your kaggle Key
GOOGLE_MAPS_API_KEY= leave this blank
```

**Step 3:** Create virtual environment called ``venv``: 
```console
python -m venv venv
```

**Step 4:** Run the script to install modules, activate virtual environment, and install packages from ``requirements.txt``:
```console
source ./leonhard_init.sh
```

**Step 5:** Open the file configs/config.yaml and insert your eth username at line 10.

**Step 6:** Train the model and produce predictions:
```console
 bsub -W 24:00 -R "rusage[mem=64000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python3 ./run.py +experiment=exp__f014
```

**Step 7:** Download the predictions from comet, you will find a file called submission.csv in the comet section called ``Assets & Artifacts``, inside the folder ``others``.


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

## Download from cluster 
The following command will download a zipped prediction into your current local folder. 
```console 
scp your_username@login.leonhard.ethz.ch:/Computational-Intelligence-Lab-2021/preds/DATE/preds.zip .
```
