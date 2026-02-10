def model_size_bytes(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * 4
