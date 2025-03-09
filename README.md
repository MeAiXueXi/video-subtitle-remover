去除了gui依赖  
修复视频针对齐BUG  
系统环境  
ubuntu:22.04  
cuda 12.4  
anconda  
torch2.5  
torchvision0.20.0  
torchaudio2.5.0  
apt install ffmpeg  



conda环境  
conda create -n videoEnv python=3.8  
conda activate videoEnv  
pip install paddlepaddle-gpu==2.6.0 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html  
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118  
pip install -r requirements.txt
