import yaml
import torch
from model.resnet_lora import ResNet
from models.yolo import Model
from loralib.utils import mark_only_lora_as_trainable
from loralib import layers as lora

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    print("--- Testing LoRA Integration into ESOD ---")

    # Đường dẫn đến file config có LoRA
    config_path = 'models/cfg/esod/visdrone_yolov5m_lora.yaml'

    # Tải cấu hình
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Khởi tạo mô hình với cấu hình LoRA
    print(f"Loading model from {config_path}...")
    model = Model(cfg=config_dict)

    print("\nModel structure check:")
    lora_found = False
    for name, module in model.named_modules():
        if isinstance(module, lora.Conv2d):
            print(f"  - Found lora.Conv2d at: {name}")
            lora_found = True
    
    if not lora_found:
        print("  - FAILED: No lora.Conv2d layers were found in the model.")
    else:
        print("  - SUCCESS: LoRA layers have been successfully injected into the model.")

    print("\nParameter count check:")
    total_params = count_parameters(model) / 1e6
    trainable_params_before = count_trainable_parameters(model) / 1e6
    print(f"  - Total parameters: {total_params:.2f}M")
    print(f"  - Trainable parameters (before freezing): {trainable_params_before:.2f}M")

    # Đóng băng các trọng số gốc, chỉ giữ lại LoRA
    mark_only_lora_as_trainable(model)

    trainable_params_after = count_trainable_parameters(model) / 1e6
    print(f"  - Trainable parameters (after freezing, LoRA only): {trainable_params_after:.2f}M")

    if trainable_params_after > 0 and trainable_params_after < total_params / 10:
        print("  - SUCCESS: The number of trainable parameters has been drastically reduced.")
    else:
        print("  - FAILED: Parameter freezing did not work as expected.")