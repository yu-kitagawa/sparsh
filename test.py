import numpy as np
import torch

from tactile_ssl.downstream_task.attentive_pooler import AttentivePooler
from tactile_ssl.model.vision_transformer import vit_base

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = vit_base(in_chans=6, pos_embed_fn="sinusoidal", num_register_tokens=1)
checkpoint = torch.load("./checkpoint/dino_vitbase.ckpt")
encoder_key = "teacher_encoder.backbone"
target_keys = [key for key in checkpoint["model"].keys() if encoder_key in key]
if "backbone" in target_keys[0] and "backbone" not in encoder_key:
    encoder_key = encoder_key + ".backbone"
new_keys = [key.replace(f"{encoder_key}.", "") for key in target_keys]
new_state_dict = {new_key: checkpoint["model"][target_key] for new_key, target_key in zip(new_keys, target_keys)}
model.load_state_dict(new_state_dict, strict=False)
model.to(device)
pooler = AttentivePooler()
pooler.to(device)

x = np.random.randn(1, 6, 224, 224)
x = x.astype(np.float32)
x = torch.from_numpy(x).to(device)

z = model(x)
output = pooler(z).squeeze()
print(output.shape)
