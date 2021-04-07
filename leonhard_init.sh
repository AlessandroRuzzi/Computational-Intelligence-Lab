module load eth_proxy gcc/6.3.0 python_gpu/3.8.5
python -m venv venv 
source venv/bin/activate
module load eth_proxy gcc/6.3.0 python_gpu/3.8.5
pip3 install -r leonhard_requirements.txt  
export PYTHONPATH=$PYTHONPATH:~/Computational-Intelligence-Lab-2021