sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
# run

                cd mistralvn
                python3 main.py 8080 cpu


# See prerequisites. Adding current user to Video and Render groups

sudo usermod -a -G render,video $LOGNAME
wget https://repo.radeon.com/amdgpu-install/6.0.3/ubuntu/jammy/amdgpu-install_6.0.60003-1_all.deb
wget https://repo.radeon.com/amdgpu-install/6.0.3/ubuntu/focal/amdgpu-install_6.0.60003-1_all.deb
sudo apt install ./amdgpu-install_6.0.60003-1_all.deb
sudo apt update
sudo apt install amdgpu-dkms,rocm
sudo apt install rocm
sudo apt install hiplibsdk
echo "Please reboot system for all settings to take effect."

https://repo.radeon.com/amdgpu-install/6.0.3/ubuntu/jammy/amdgpu-install_6.0.60003-1_all.deb

sudo apt install ./amdgpu-install_6.0.60003-1_all.deb

sudo amdgpu-install --usecase=hiplibsdk,rocmdev,dkms,mllib,mlsdk 

https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/prerequisites.html

sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"

# LLM model


                https://oauth:...your token..@huggingface.co/Viet-Mistral/Vistral-7B-Chat

# XNDA amd npu

                https://github.com/amd/xdna-driver

                git clone -b iommu_sva_part4_v6_v6.8_rc2 https://github.com/AMDESE/linux.git