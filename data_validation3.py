import torch

def summarize_tensor(name, tensor):
    try:
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        mean_val = tensor.mean().item()
        print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
              f"min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}")
    except Exception as e:
        print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
              f"(stats unavailable: {e})")

def inspect_pt_file(file_path):
    print(f"Loading: {file_path}")
    data = torch.load(file_path, map_location="cpu")

    def recursive_check(obj, prefix=""):
        if isinstance(obj, torch.Tensor):
            summarize_tensor(prefix or "tensor", obj)

        elif isinstance(obj, dict):
            for k, v in obj.items():
                recursive_check(v, f"{prefix}.{k}" if prefix else k)

        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                recursive_check(v, f"{prefix}[{i}]")

        else:
            pass

    recursive_check(data)

if __name__ == "__main__":
    inspect_pt_file("../Data_NLPinHealth/tokenized_phenotyping/val_tokenized.pt")
