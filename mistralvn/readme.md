sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
# See prerequisites. Adding current user to Video and Render groups
sudo usermod -a -G render,video $LOGNAME
wget https://repo.radeon.com/amdgpu-install/6.0.3/ubuntu/jammy/amdgpu-install_6.0.60003-1_all.deb
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