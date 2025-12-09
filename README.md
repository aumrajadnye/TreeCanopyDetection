# TreeCanopyDetection
GitHub repository for the [Solafune Tree Canopy Detection Hackathon.](https://solafune.com/competitions/26ff758c-7422-4cd1-bfe0-daecfc40db70?menu=about&tab=).

inclusion of scene_prediction

steps: 
- run main.py first
- in terminal:
    
.venv/Scripts/activate
    
$env:PYTHONPATH="C:/Users/keini/OneDrive/Desktop/Code/TreeCanopyDetection"

python ../mmsegmentation/tools/train.py configs/trcnpy_cnfg.py

For GPU Before installation of mmcv direct path to location of GPU using the following lines in the terminal :
- setx CUDA_HOME "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
- setx PATH "$env:CUDA_HOME\bin;$env:CUDA_HOME\libnvvp;$env:PATH"


py -3.10 -m venv .venv

.venv/Scripts/activate

$env:PATH=""

$env:PATH = "C:\Users\harsh\Desktop\Keinisha\TreeCanopyDetection\.venv\Scripts;$env:PATH"

$env:PATH = "C:\Windows\System32;C:\Windows;C:\Windows\System32\WindowsPowerShell\v1.0\" + ";" + $env:PATH

$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin;$env:PATH"

$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"

$env:PATH = "C:\Program Files\Git\cmd;$env:PATH"

set PYTHONPATH=C:\Users\harsh\Desktop\Keinisha\mmsegmentation

$env:PYTHONPATH="C:\Users\harsh\Desktop\Keinisha\TreeCanopyDetection"

$env:VSINSTALLDIR="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\"

$env:VCINSTALLDIR="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\"

% $env:PATH="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;$env:PATH"

$env:MAX_JOBS=1

set MMCV_WITH_OPS=1

$env:TORCH_CUDA_ARCH_LIST="8.6"

cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && set'

python -m pip install --upgrade pip 

pip install ftfy

pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

pip install mmengine==0.7.4

pip install mmsegmentation==0.30.0

pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

pip uninstall numpy

pip install numpy<2

pip freeze > requirements_mmseg.txt

python ../mmsegmentation/tools/train.py configs/trcnpy_cnfg.py












<!-- Does not work for windows -->
<!-- git clone https://github.com/open-mmlab/mmcv.git  

cd mmcv  

pip install -r requirements/optional.txt
 
pip install -r requirements/runtime.txt                   

set FORCE_NVCC_FLAGS=-allow-unsupported-compiler

python setup.py build_ext --inplace -v -->






<!-- $env:INCLUDE += ";C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\ucrt"

$env:LIB += ";C:\Program Files (x86)\Windows Kits\10\Lib\10.0.19041.0\ucrt\x64" -->
