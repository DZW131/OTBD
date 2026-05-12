# 使用官方 Miniconda 基础镜像（Python 3.12）
FROM continuumio/miniconda3:latest

# 设置作者信息（可选）
LABEL maintainer="simon"

# 创建工作目录
WORKDIR /workspace

# 复制所需文件，要是全部复制就 [COPY . .]，后面的.代表/workspace，也就是定义的工作目录
# COPY environment.yml .
COPY DATASET/ ./DATASET/
COPY run_infer.sh .
# COPY DATASET/mT_Dataset.py  /tmp/mT_Dataset.py
# COPY DATASET/nnUNetTrainer.py  /tmp/nnUNetTrainer.py

# 使用 conda 创建环境（手动指定）或基于 environment.yml
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free && \
    conda config --set show_channel_urls yes
# RUN conda create -n dockertest python=3.12 && \
#     conda clean -afy

# 创建 Conda 环境
# 设置 pip 国内源（修复你的错误拼写）
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# RUN conda env create -f environment.yml
# RUN conda env create -f slim-env.yml

# 创建并激活 Conda 环境，安装依赖
# RUN conda update -n base -c defaults conda -y
RUN conda create -n nnunet_docker python=3.11 -y
RUN conda run -n nnunet_docker pip install --upgrade pip setuptools
RUN conda run -n nnunet_docker pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126
RUN conda run -n nnunet_docker pip install --no-cache-dir nnunetv2 torchio


# 克隆 nnUNet 项目并安装（注意放在容器中）
RUN git clone https://github.com/MIC-DKFZ/nnUNet.git
RUN conda run -n nnunet_docker pip install -e ./nnUNet

# 激活环境（注意：Docker中无法持久激活，需手动指定）
SHELL ["conda", "run", "-n", "nnunet_docker", "/bin/bash", "-c"]


# 启动命令示例
# export nnUNet_raw="./workspace/DATASET/nnUNet_raw"
# export nnUNet_preprocessed="./workspace/DATASET/nnUNet_preprocessed"
# export nnUNet_results="./workspace/DATASET/nnUNet_result"

ENV nnUNet_raw="/workspace/DATASET/nnUNet_raw"
ENV nnUNet_preprocessed="/workspace/DATASET/nnUNet_preprocessed"
ENV nnUNet_results="/workspace/DATASET/nnUNet_result"

# CMD ["conda", "run", "-n", "dockertest", "python", "main.py"]

# 假设你已经安装了 nnunetv2 包
# 复制文件到包目录中
# 拷贝你的自定义文件到临时位置

# 复制到安装路径 mT_Dataset.py
RUN conda run -n nnunet_docker python -c "\
import nnunetv2, shutil; \
shutil.copy('/workspace/DATASET/mT_Dataset.py', nnunetv2.__path__[0] + '/training/dataloading/mT_Dataset.py')"

# 复制到安装路径 cotta_wrapper.py
RUN conda run -n nnunet_docker python -c "\
import nnunetv2, shutil; \
shutil.copy('/workspace/DATASET/cotta_wrapper.py', nnunetv2.__path__[0] + '/inference/cotta_wrapper.py')"

## 删除原来的 nnUNetTrainer.py
RUN conda run -n nnunet_docker python -c "\
import os, nnunetv2; \
target = os.path.join(nnunetv2.__path__[0], 'training', 'nnUNetTrainer', 'nnUNetTrainer.py'); \
print(f'Deleting: {target}'); \
os.remove(target); \
print(f'Deleted over: {target}')"

## 添加新的 nnUNetTrainer.py
RUN conda run -n nnunet_docker python -c "\
import nnunetv2, shutil; \
shutil.copy('/workspace/DATASET/nnUNetTrainer.py', nnunetv2.__path__[0] + '/training/nnUNetTrainer/nnUNetTrainer.py')"

## 删除原来的 predict_from_raw_data.py
RUN conda run -n nnunet_docker python -c "\
import os, nnunetv2; \
target = os.path.join(nnunetv2.__path__[0], 'inference', 'predict_from_raw_data.py'); \
print(f'Deleting: {target}'); \
os.remove(target); \
print(f'Deleted over: {target}')"

## 添加新的 predict_from_raw_data.py
RUN conda run -n nnunet_docker python -c "\
import nnunetv2, shutil; \
shutil.copy('/workspace/DATASET/predict_from_raw_data.py', nnunetv2.__path__[0] + '/inference/predict_from_raw_data.py')"

# COPY DATASET/mT_Dataset.py /opt/conda/envs/nnunet-env/lib/python3.11/site-packages/nnunetv2/training/dataloading
# RUN rm -f /opt/conda/envs/nnunet-env/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer
# COPY DATASET/nnUNetTrainer.py /opt/conda/envs/nnunet-env/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer

# 启动时执行 entrypoint 脚本
CMD ["conda", "run", "-n", "nnunet_docker", "/bin/bash", "/workspace/run_infer.sh"]