# PAPER BASED EVALUATOR KRAEPELIN
repository kode sistem untuk skripsi dari Hadekha dan Christian terkait sistem evaluasi Kraepelin berbasis kertas berbasis computer vision dan digit
recognition

## INFERENCE ENGINE
menggunakan pretrained model CNN yang dievaluasi menggunakan dataset MNIST dan EMNIST serta
dengan tambahan data augmentation. Setiap sampel training set telah diaugmentasi (zoom, shift, rotate) untuk menambah
akurasi dari model.
credit to : https://www.kaggle.com/models/pauljohannesaru/beyond_mnist

## STRUKTUR KODE

### dataservices
merepresentasikan kode terkait abstraksi pengelolaan dan pengambilan data

### controllers
merepresentasikan kode logic / function business process dari system

### models
merepresentasikan abstraksi objek / entitas untuk mempermudah abstraksi system

### engines
merepresentasikan mesin utama dari system seperti inference engine, preprocessing, dan region of interest

### persistent
direktori aset business process disimpan, seperti upload documents, rekam jejak evaluasi, dan gambar hasil teknik region of interest

## DISCLAIMER

this code is built with monolithic architecture in mind



### DEPLOY IT ON YOUR MACHINE!

## Prerequisites

### 1. Install Docker

#### Windows:
1. Download Docker Desktop from [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
2. Run the installer and follow the setup wizard
3. Restart your computer when prompted
4. Open Docker Desktop and complete the initial setup
5. Verify installation by opening Command Prompt or PowerShell and running:
   ```cmd
   docker --version
   ```

#### macOS:
1. Download Docker Desktop for Mac from [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
2. Open the downloaded `.dmg` file and drag Docker to Applications
3. Launch Docker from Applications folder
4. Complete the initial setup process
5. Verify installation by opening Terminal and running:
   ```bash
   docker --version
   ```

#### Linux (Ubuntu/Debian):
1. Update your package index:
   ```bash
   sudo apt update
   ```
2. Install required packages:
   ```bash
   sudo apt install apt-transport-https ca-certificates curl gnupg lsb-release
   ```
3. Add Docker's official GPG key:
   ```bash
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
   ```
4. Add Docker repository:
   ```bash
   echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   ```
5. Install Docker:
   ```bash
   sudo apt update
   sudo apt install docker-ce docker-ce-cli containerd.io
   ```
6. Start and enable Docker:
   ```bash
   sudo systemctl start docker
   sudo systemctl enable docker
   ```
7. Add your user to docker group (optional, to run without sudo):
   ```bash
   sudo usermod -aG docker $USER
   ```
   Log out and back in for this to take effect.
8. Verify installation:
   ```bash
   docker --version
   ```

#### Linux (CentOS/RHEL/Fedora):
1. Install required packages:
   ```bash
   sudo dnf install -y dnf-plugins-core
   ```
2. Add Docker repository:
   ```bash
   sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
   ```
3. Install Docker:
   ```bash
   sudo dnf install docker-ce docker-ce-cli containerd.io
   ```
4. Start and enable Docker:
   ```bash
   sudo systemctl start docker
   sudo systemctl enable docker
   ```
5. Verify installation:
   ```bash
   docker --version
   ```

### 2. Install Docker Compose

#### Windows & macOS:
Docker Compose is included with Docker Desktop, so no additional installation is needed.

Verify by running:
```bash
docker compose version
```

#### Linux:
Docker Compose is also included with modern Docker installations. If not available, install it manually:

1. Download the latest version:
   ```bash
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   ```
2. Make it executable:
   ```bash
   sudo chmod +x /usr/local/bin/docker-compose
   ```
3. Verify installation:
   ```bash
   docker-compose --version
   ```

**Note:** Some newer installations use `docker compose` (space) instead of `docker-compose` (hyphen).

3. Clone this repository

make sure you have a steady internet connection

paste and execute this command on your terminal / git bash / command prompt

```bash
git clone https://github.com/Skripsi-Kraepelin-Paper-Evaluation/digit-recognition-engine
```

4. Run docker compose

make sure port 8081 and 8080 is not used

walk / change towards cloned project repository

```bash
cd digit-recognition-engine
```

execute this commands

```bash
docker compose up -d
```

5. Open your browser and try it on

open your favorite browser and open
http://localhost:8081

6. Verify the installation

Check if all containers are running properly:

```bash
docker compose ps
```

You should see all services up and running. If any service shows as "exited" or "restarting", check the logs:

```bash
docker compose logs [service-name]
```

7. Stopping the application

When you're done using the application, you can stop all services:

```bash
docker compose down
```

To also remove the volumes and clean up completely:

```bash
docker compose down -v
```

## Troubleshooting

### Common Issues:

**Port conflicts:**
- If ports 8080 or 8081 are already in use, modify the `docker-compose.yml` file to use different ports
- Change the port mapping from `8081:8081` to `[available-port]:8081`

**Docker permission issues (Linux/Mac):**
```bash
sudo docker compose up -d
```

**Build issues:**
If you encounter build errors, try rebuilding the containers:
```bash
docker compose up --build -d
```

**Network connectivity:**
Ensure Docker daemon is running and you have internet access for pulling images.

### System Requirements:
- Docker Engine 20.10 or higher
- Docker Compose 2.0 or higher
- At least 2GB free RAM
- 5GB free disk space

### Additional Commands:

**View real-time logs:**
```bash
docker compose logs -f
```

**Access container shell:**
```bash
docker compose exec [service-name] /bin/bash
```

**Update the application:**
```bash
git pull origin main
docker compose down
docker compose up --build -d
```