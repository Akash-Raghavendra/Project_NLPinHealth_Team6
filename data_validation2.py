import torch

data = torch.load("../Data_NLPinHealth/tokenized_phenotyping/train_tokenized.pt", map_location="cpu")

def contains_text(obj):
    if isinstance(obj, str):
        return True
    elif isinstance(obj, dict):
        return any(contains_text(v) for v in obj.values())
    elif isinstance(obj, (list, tuple)):
        return any(contains_text(v) for v in obj)
    elif isinstance(obj, torch.Tensor):
        return False
    return False

print("Contains text:", contains_text(data))
