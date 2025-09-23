# hpr_3dgs
slay

##



### Setup conda requirements

```bash
# GPU Installation on a CUDA 11.6 Machine
conda create -n hpr_splats python=3.10
conda activate hpr_splats
pip install torch --index-url https://download.pytorch.org/whl/cu118 # Modify according to your cuda version. For example, cu121 for CUDA 12.1
pip install fvcore iopath
MAX_JOBS=8 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" # This will take some time to compile!
pip install -r requirements.txt

# CPU Installation
conda create -n hpr_splats python=3.10
conda activate hpr_splats
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fvcore iopath
MAX_JOBS=8 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install -r requirements.txt

```