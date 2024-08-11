import torch
from thop import profile
from models import HCRM, HSRM
from importlib_metadata import version

### FLOPS ANALYSIS ###

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HCRM().to(device)
model.load_state_dict(torch.load('./model/handwritten_character_recognition_model.pth'))

print("thop version:", version("thop"))

# MACS = multi-accumulate operations (typically counted as 2 flops)
# ->> one multiply and one accumulate
input_tensor = torch.randn(1, 1, 28, 28).to(device)
macs, params = profile(model, inputs=(input_tensor,), verbose=False)
flops = 2*macs
print(f"{flops:.1e} FLOPS")

del model
torch.cuda.empty_cache()