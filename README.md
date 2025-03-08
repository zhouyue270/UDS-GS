# Unbounded Dynamic Scene Reconstruction via 3D Gaussian Splatting: An Efficient Approach for Real-Time Rendering

## 📌Introduction
UDS-GS is an advanced **3D Gaussian-based dynamic scene reconstruction** method designed for **autonomous driving, robotics, and virtual reality**. It efficiently reconstructs unbounded dynamic scenes in real-time by integrating **LiDAR and Structure-from-Motion (SFM) point clouds**, enhancing geometric precision. The model leverages **Gaussian color feature prediction networks** to effectively capture **local and global feature information**, ensuring high-quality scene rendering.  

## 📌 Pipeline Overview  
Below is the pipeline of our **UDS-GS** framework:  

![Pipeline Overview](https://your-image-url.com/pipeline.png)

## 📂Datasets
We use the following datasets for training and evaluation:
- **Waymo Open Dataset**: [https://waymo.com/open/](https://waymo.com/open/)
- **KITTI Dataset**: [https://www.cvlibs.net/datasets/kitti/](https://www.cvlibs.net/datasets/kitti/)

## 1️⃣Installation
Follow these steps to set up the environment:

```bash
# Clone this repository
git clone https://github.com/zhouyue270/UDS-GS.git
cd UDS-GS

# Create a conda environment
conda create -n UDS-GS python=3.8 -y
conda activate UDS-GS

# Install dependencies
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt

# Clone and install additional dependencies
git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
pip install ./submodules/simple-waymo-open-dataset-reader
```

## 2️⃣Training
To train the model, run:
```bash
python train.py --config configs/training/xxxx.yaml
```

## 3️⃣Validation
To validate the trained model, run:
```bash
python train.py --config configs/validation/xxxx.yaml
```

## 4️⃣Rendering
To render the trained model, use:
```bash
python render.py --config configs/training/xxxx.yaml mode evaluate
python render.py --config configs/validation/xxxx.yaml mode evaluate
```

## 5️⃣Visualization
To visualize the results, install the dependencies and use the provided viewer:

### Install dependencies
```bash
cd SIBR_viewers
cmake -Bbuild .
cmake --build build --target install --config RelWithDebInfo

sudo apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev \
    libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev \
    libxxf86vm-dev libembree-dev
```

### Build and run the viewer
```bash
cd SIBR_viewers
cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release # add -G Ninja to build faster
cmake --build build -j24 --target install 

./<SIBR install dir>/bin/SIBR_gaussianViewer_app -m <path to trained model>
