# Running the Image Classification Notebook on an Ubuntu 24.04 VPS

This guide covers setting up JupyterHub on Ubuntu 24.04 LTS and running `image_class.ipynb` in two scenarios:

1. **CPU-only VPS** (no GPU)
2. **GPU VPS** with an NVIDIA RTX 4000 and 16 GB VRAM

---

## Part 1: CPU-Only VPS Setup

### 1.1 System Prerequisites

```bash
# Update the system
sudo apt update && sudo apt upgrade -y

# Install core dependencies
sudo apt install -y python3 python3-pip python3-venv python3-dev \
    build-essential curl wget git npm nodejs openjdk-17-jre-headless
```

Java is required for Apache Kafka, which the notebook uses as its message broker.

### 1.2 Install JupyterHub

```bash
# Install configurable-http-proxy (JupyterHub dependency)
sudo npm install -g configurable-http-proxy

# Create a Python virtual environment for JupyterHub
sudo python3 -m venv /opt/jupyterhub
sudo /opt/jupyterhub/bin/pip install wheel
sudo /opt/jupyterhub/bin/pip install jupyterhub jupyterlab notebook
```

### 1.3 Configure JupyterHub

```bash
sudo mkdir -p /etc/jupyterhub
cd /etc/jupyterhub
sudo /opt/jupyterhub/bin/jupyterhub --generate-config
```

Edit `/etc/jupyterhub/jupyterhub_config.py` and set at minimum:

```python
c.JupyterHub.bind_url = 'http://0.0.0.0:8000'
c.Spawner.default_url = '/lab'
c.JupyterHub.admin_access = True
# Add your username as admin
c.Authenticator.admin_users = {'your_username'}
```

### 1.4 Create a Systemd Service for JupyterHub

Create `/etc/systemd/system/jupyterhub.service`:

```ini
[Unit]
Description=JupyterHub
After=network.target

[Service]
User=root
Environment="PATH=/opt/jupyterhub/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/opt/jupyterhub/bin/jupyterhub -f /etc/jupyterhub/jupyterhub_config.py
WorkingDirectory=/etc/jupyterhub
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable jupyterhub
sudo systemctl start jupyterhub
```

JupyterHub is now running on port 8000. Log in with your Linux user credentials.

### 1.5 Install Python Dependencies for the Notebook

Each JupyterHub user needs the packages in their own environment. Log in as your user and run:

```bash
pip install tensorflow keras numpy matplotlib kafka-python pillow tqdm
```

> **Note:** On a CPU-only machine, `pip install tensorflow` installs the CPU-only build automatically on version 2.16+. For older versions, use `pip install tensorflow-cpu`.

### 1.6 Install and Configure Apache Kafka

The notebook tries to download and start Kafka inline. For a VPS it is better to set it up as a service.

```bash
# Download Kafka
cd /opt
sudo wget https://archive.apache.org/dist/kafka/3.5.0/kafka_2.13-3.5.0.tgz
sudo tar -xzf kafka_2.13-3.5.0.tgz
sudo mv kafka_2.13-3.5.0 kafka
sudo rm kafka_2.13-3.5.0.tgz
```

#### Zookeeper Service

Create `/etc/systemd/system/zookeeper.service`:

```ini
[Unit]
Description=Apache Zookeeper
After=network.target

[Service]
Type=simple
User=root
ExecStart=/opt/kafka/bin/zookeeper-server-start.sh /opt/kafka/config/zookeeper.properties
ExecStop=/opt/kafka/bin/zookeeper-server-stop.sh
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

#### Kafka Service

Create `/etc/systemd/system/kafka.service`:

```ini
[Unit]
Description=Apache Kafka
After=zookeeper.service
Requires=zookeeper.service

[Service]
Type=simple
User=root
ExecStart=/opt/kafka/bin/kafka-server-start.sh /opt/kafka/config/server.properties
ExecStop=/opt/kafka/bin/kafka-server-stop.sh
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable zookeeper kafka
sudo systemctl start zookeeper kafka
```

Verify Kafka is running:

```bash
/opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list
```

### 1.7 Notebook Modifications for CPU-Only VPS

Before running, make these adjustments in the notebook:

#### 1. Skip the Kafka install/start cells (Steps 1 and 9)

Since Kafka is already running as a system service, **comment out or skip** the cells that download, start, and restart Kafka. These are:

- Cell 2 (Step 1): `!pip install ...`, `!curl`, Zookeeper/Kafka start commands
- Cell 19 (Step 9): The Kafka restart cell

Keep only the `pip install` lines if you haven't installed the packages yet.

#### 2. Reduce training time

CPU training is roughly 10-20x slower than GPU. Edit the training cells (Steps 3 and 4) to reduce the workload:

```python
# In both CNN and ResNet training cells, change:
epochs=50
# To:
epochs=15

# Optionally reduce the test set in the Judge agent (Step 7):
# Change num_images=500 to num_images=100 for faster runs
```

Early stopping will likely terminate training before hitting the epoch limit anyway.

#### 3. Increase Kafka consumer timeout

CPU inference is slower, so the consumers may time out. In the agent cells (Steps 5 and 6), increase the timeout:

```python
# Change:
consumer_timeout_ms=60000
# To:
consumer_timeout_ms=300000  # 5 minutes
```

Also increase the Judge agent's consumer timeout (Step 7):

```python
# Change:
consumer_timeout_ms=10000
# To:
consumer_timeout_ms=120000  # 2 minutes
```

### 1.8 Expected CPU Performance

| Metric | Approximate Value |
|---|---|
| CNN training (15 epochs) | 15-30 minutes |
| ResNet training (15 epochs) | 20-40 minutes |
| Inference (500 images, both models) | 2-5 minutes |
| Total wall time | 40-75 minutes |

Times vary based on CPU cores and clock speed. A 4-core VPS at 2.5 GHz is assumed above.

### 1.9 Run the Notebook

1. Open JupyterHub at `http://your-vps-ip:8000`
2. Upload `image_class.ipynb`
3. Skip cells 2 and 19 (Kafka setup/restart)
4. Run the remaining cells in order

---

## Part 2: GPU VPS Setup (RTX 4000, 16 GB VRAM)

Start with all the steps from Part 1 (JupyterHub, Kafka, Python packages). Then add GPU support.

### 2.1 Install NVIDIA Drivers

```bash
# Check if a driver is already installed
nvidia-smi

# If not, install the recommended driver
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot
```

After reboot, verify:

```bash
nvidia-smi
```

You should see the RTX 4000 listed with driver version and CUDA version.

### 2.2 Install CUDA Toolkit

TensorFlow 2.19 requires CUDA 12.x and cuDNN 9.x.

```bash
# Install CUDA 12 toolkit
sudo apt install -y nvidia-cuda-toolkit

# Verify
nvcc --version
```

If the system package is outdated, install from NVIDIA's repository instead:

```bash
# Add NVIDIA's package repository (for CUDA 12.4)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4
```

Add to your shell profile (`~/.bashrc`):

```bash
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```

### 2.3 Install cuDNN

```bash
sudo apt install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12
```

Or download from NVIDIA's cuDNN archive if the package is not available.

### 2.4 Install TensorFlow with GPU Support

```bash
# Uninstall CPU-only tensorflow if previously installed
pip uninstall -y tensorflow

# Install GPU-enabled tensorflow
pip install tensorflow[and-cuda]
```

Verify GPU is detected:

```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Expected output:

```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### 2.5 Configure TensorFlow Memory Growth

The RTX 4000 has 16 GB of VRAM. By default TensorFlow allocates all of it. To avoid memory issues when running two models concurrently, add this cell at the top of the notebook (after imports):

```python
import tensorflow as tf

# Prevent TensorFlow from grabbing all GPU memory at once
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU detected: {gpus[0].name}")
else:
    print("No GPU detected, running on CPU")
```

### 2.6 Notebook Modifications for GPU VPS

#### 1. Skip Kafka install/start cells

Same as Part 1 -- skip cells 2 and 19 since Kafka runs as a system service.

#### 2. Keep full training parameters

With the RTX 4000, you can run the notebook at full capacity:

- Keep `epochs=50` (early stopping will manage it)
- Keep `num_images=500` or increase it
- Keep default consumer timeouts (GPU inference is fast)

#### 3. Optional: increase batch size for faster training

Edit the training cells to take advantage of 16 GB VRAM:

```python
# Change batch_size in both CNN and ResNet training cells:
batch_size=128   # up from 64
```

CIFAR-10 images are small (32x32x3), so even `batch_size=256` should fit comfortably.

### 2.7 Expected GPU Performance

| Metric | Approximate Value |
|---|---|
| CNN training (50 epochs, early stop ~15) | 2-4 minutes |
| ResNet training (50 epochs, early stop ~15) | 3-6 minutes |
| Inference (500 images, both models) | 10-30 seconds |
| Total wall time | 5-12 minutes |

### 2.8 Run the Notebook

1. Open JupyterHub at `http://your-vps-ip:8000`
2. Upload `image_class.ipynb`
3. Add the memory growth cell at the top
4. Skip cells 2 and 19 (Kafka setup/restart)
5. Run the remaining cells in order

---

## Troubleshooting

### Kafka won't start

```bash
# Check logs
sudo journalctl -u kafka -n 50
sudo journalctl -u zookeeper -n 50

# Make sure Java is installed
java -version

# Clear stale data and restart
sudo rm -rf /tmp/kafka-logs /tmp/zookeeper
sudo systemctl restart zookeeper kafka
```

### TensorFlow doesn't see the GPU

```bash
# Verify driver
nvidia-smi

# Check CUDA
nvcc --version

# Check cuDNN
dpkg -l | grep cudnn

# Test in Python
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Common fixes:
- Reboot after driver install
- Ensure `LD_LIBRARY_PATH` includes `/usr/local/cuda/lib64`
- Ensure TensorFlow version matches CUDA version (TF 2.19 needs CUDA 12.x)

### Kafka consumer timeout errors

If agents time out waiting for messages, increase `consumer_timeout_ms` in the notebook. On CPU this is especially likely during inference of 500 images.

### JupyterHub login fails

```bash
# Make sure your Linux user has a password set
sudo passwd your_username

# Check JupyterHub logs
sudo journalctl -u jupyterhub -n 50
```

### Out of memory (GPU)

If you see CUDA OOM errors, add the memory growth config from section 2.5, or reduce `batch_size` to 32.

---

## Firewall Notes

If your VPS has a firewall (ufw, iptables, or cloud security group), open port 8000 for JupyterHub:

```bash
sudo ufw allow 8000/tcp
```

For production use, put JupyterHub behind a reverse proxy (nginx) with HTTPS instead of exposing port 8000 directly.
