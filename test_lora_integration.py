import yaml
import torch
import sys
from pathlib import Path

# Đảm bảo các mô-đun trong dự án có thể được import
# Thêm thư mục gốc của dự án vào sys.path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Thư mục gốc của dự án
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.yolo import Model
from loralib.utils import mark_only_lora_as_trainable
from loralib import layers as lora
from utils.general import colorstr

def count_parameters(model):
    """Đếm tổng số tham số trong mô hình."""
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    """Đếm số lượng tham số có thể huấn luyện (requires_grad=True)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_test():
    """Chạy toàn bộ quy trình kiểm tra."""
    print(colorstr('bold', '--- Bắt đầu kiểm tra tích hợp LoRA vào ESOD ---'))

    # 1. Định nghĩa đường dẫn đến tệp cấu hình LoRA
    config_path = 'models/cfg/esod/visdrone_yolov5m_lora_test.yaml'
    print(f"\n1. Tải cấu hình từ: {config_path}")
    
    if not Path(config_path).exists():
        print(colorstr('red', f"LỖI: Tệp cấu hình '{config_path}' không tồn tại. Vui lòng tạo tệp này trước."))
        return

    # 2. Khởi tạo mô hình từ tệp cấu hình
    try:
        print("   Đang khởi tạo mô hình...")
        # Chúng ta cần truyền config_dict trực tiếp vào Model constructor
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Đặt nc và anchors mặc định nếu cần để tránh lỗi
        config_dict.setdefault('nc', 80)
        config_dict.setdefault('anchors', [
            [10,13, 16,30, 33,23],
            [30,61, 62,45, 59,119],
            [116,90, 156,198, 373,326]
        ])
        
        model = Model(cfg=config_dict, ch=3)
        print(colorstr('green', "   Khởi tạo mô hình thành công!"))
    except Exception as e:
        print(colorstr('red', f"LỖI: Không thể khởi tạo mô hình. Lỗi: {e}"))
        return

    # 3. Kiểm tra cấu trúc mô hình để xác nhận các lớp LoRA đã được chèn
    print("\n2. Kiểm tra cấu trúc mô hình...")
    lora_layers_found = []
    for name, module in model.named_modules():
        # Kiểm tra lớp Conv2d bên trong lora.Conv2d
        if isinstance(module, lora.Conv2d):
            lora_layers_found.append(name)
    
    if not lora_layers_found:
        print(colorstr('red', "   THẤT BẠI: Không tìm thấy lớp lora.Conv2d nào trong mô hình."))
        print("   Vui lòng kiểm tra lại logic trong `parse_model` và `Conv.__init__`.")
        return
    else:
        print(colorstr('green', f"   THÀNH CÔNG: Tìm thấy {len(lora_layers_found)} lớp lora.Conv2d."))
        print(f"   Ví dụ: '{lora_layers_found[0]}'")

    # 4. Kiểm tra số lượng tham số trước và sau khi đóng băng
    print("\n3. Kiểm tra số lượng tham số...")
    total_params = count_parameters(model)
    trainable_before = count_trainable_parameters(model)
    
    print(f"   - Tổng số tham số: {total_params/1e6:.2f}M")
    if total_params == trainable_before:
        print(f"   - Số tham số có thể huấn luyện (trước khi đóng băng): {trainable_before/1e6:.2f}M (Đúng)")
    else:
        print(colorstr('yellow', f"   - CẢNH BÁO: Số tham số có thể huấn luyện ({trainable_before/1e6:.2f}M) không bằng tổng số tham số."))

    # Đóng băng các trọng số gốc, chỉ giữ lại các tham số của LoRA
    print("   Đang đóng băng các trọng số không phải LoRA...")
    mark_only_lora_as_trainable(model, bias='none') # 'bias=none' để chỉ huấn luyện ma trận A, B

    trainable_after = count_trainable_parameters(model)
    print(f"   - Số tham số có thể huấn luyện (chỉ LoRA): {trainable_after/1e3:.1f}K ({trainable_after})")
    
    # Kiểm tra logic
    if 0 < trainable_after < total_params * 0.05: # Số tham số lora phải nhỏ hơn 5% tổng số
        print(colorstr('green', "   THÀNH CÔNG: Số lượng tham số có thể huấn luyện đã giảm đáng kể."))
    else:
        print(colorstr('red', "   THẤT BẠI: Việc đóng băng tham số không hoạt động như mong đợi."))
        print(f"     - Tỉ lệ tham số huấn luyện: {trainable_after/total_params:.2%}")
        return
        
    print("\n4. Kiểm tra các tham số có thể huấn luyện:")
    found_lora_param = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"   - {name} (kích thước: {list(param.shape)}) -> Cần huấn luyện")
            if 'lora_' in name:
                found_lora_param = True
    
    if found_lora_param:
        print(colorstr('green', "   THÀNH CÔNG: Các tham số có thể huấn luyện đều là tham số của LoRA."))
    else:
        print(colorstr('red', "   THẤT BẠI: Không tìm thấy tham số LoRA nào trong danh sách cần huấn luyện."))

    print(colorstr('bold', '\n--- Hoàn tất kiểm tra ---'))
    print(colorstr('green', 'Tích hợp LoRA có vẻ đã thành công. Sẵn sàng cho Giai đoạn 2!'))

if __name__ == '__main__':
    run_test()