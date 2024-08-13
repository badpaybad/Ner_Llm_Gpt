import torch
from pytorch_pretrained_gans import make_gan
#pip install boto3 --user
#Downloading: "http://selfcondgan.csail.mit.edu/weights/selfcondgan_i_model.pt" to /home/dunp/.cache/torch/hub/checkpoints/selfcondgan_i_model.pt
# SelfCondGAN (conditional)
G = make_gan(gan_type='selfconditionedgan', model_name='self_conditioned')
# G = make_gan(gan_type='biggan', model_name='biggan-deep-256')  # -> nn.Module
y = G.sample_class(batch_size=1)  # -> torch.Size([1, 1000])
z = G.sample_latent(batch_size=1)  # -> torch.Size([1, 128])
print(y)
print(z)
x = G(z=z, y=y)  # -> torch.Size([1, 3, 256, 256])
print(x)
# assert list(x.shape) == [1, 3, 256, 256]
import cv2

# frameorg= cv2.imread("/work/llm/Ner_Llm_Gpt/deepfacefake/areafake.png")
# framefake= cv2.imread("/work/llm/Ner_Llm_Gpt/deepfacefake/keeped.png")

# h,w,c= frameorg.shape

# framefake=cv2.resize(framefake, (w,h))

# frameorg = cv2.cvtColor(frameorg, cv2.COLOR_BGRA2BGR)
# framefake = cv2.cvtColor(framefake, cv2.COLOR_BGRA2BGR)
# x = G(z=torch.from_numpy( frameorg).clone().int(), y=torch.from_numpy( framefake).clone().int()) 

# cv2.imshow("", x.to("cpu"))
# cv2.waitKey(0)