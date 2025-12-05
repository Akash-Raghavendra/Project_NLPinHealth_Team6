import torch

def inspect_pt_file(file_path):
    obj = torch.load(file_path, map_location="cpu")
    print(f"Type of object: {type(obj)}")

    if isinstance(obj, dict):
        print("Top-level keys:", obj.keys())
        suspicious_keys = []
        for k, v in obj.items():
            if isinstance(v, torch.Tensor):
                print(f"Tensor key: {k}, shape: {v.shape}")
                if v.dim() > 2 and v.numel() < 100000:
                    suspicious_keys.append(k)
        if suspicious_keys:
            print("These tensors could potentially be raw data:", suspicious_keys)
        else:
            print("No raw dataset tensors detected.")

    elif isinstance(obj, torch.Tensor):
        print(f"Tensor shape: {obj.shape}, num elements: {obj.numel()}")
        if obj.numel() < 100000:
            print("This tensor could potentially be raw dataset values.")
        else:
            print("Likely model weights or embeddings.")

    else:
        print("Unknown object type — inspect manually:", type(obj))

file_path_train = "../Data_NLPinHealth/tokenized_phenotyping/train_tokenized.pt"
file_path_test = "../Data_NLPinHealth/tokenized_phenotyping/test_tokenized.pt"
file_path_val = "../Data_NLPinHealth/tokenized_phenotyping/val_tokenized.pt"

inspect_pt_file(file_path_train)
inspect_pt_file(file_path_test)
inspect_pt_file(file_path_val)

