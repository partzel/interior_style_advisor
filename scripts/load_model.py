import torch
import json
from huggingface_hub import hf_hub_download
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class ClipProjector(nn.Module):
  def __init__(self, in_dim, out_dim, seq_len=1):
    super().__init__()
    self.linear = nn.Linear(in_dim, out_dim * seq_len)
    self.seq_len = seq_len
    self.out_dim = out_dim

  def forward(self, x):
    x = self.linear(x)
    return x.view(-1, self.seq_len, self.out_dim)

def load_projector(repo_id="sameddallaa/projector-clip-llm", subfolder="projector"):
  config_path = hf_hub_download(repo_id=repo_id, filename="config.json", subfolder=subfolder)
  weights_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin", subfolder="projector")
  with open(config_path) as f:
    config = json.load(f)
    
    model = ClipProjector(
      in_dim=config["in_dim"],
      out_dim=config["out_dim"],
      seq_len=config["seq_len"],
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model.to(device).eval()