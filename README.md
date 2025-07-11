适配昇腾NPU的ultralytics
（仅在昇腾910b上测试）
python环境：Python 3.10.18

pip install ultralytics==8.3.161

pip uninstall torch torchvision

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
 
pytorch_npu可参考https://www.hiascend.com/document/detail/zh/Pytorch/700/configandinstg/instg/insg_0004.html

wget https://gitee.com/ascend/pytorch/releases/download/v7.0.0-pytorch2.1.0/torch_npu-2.1.0.post12-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

pip install torch_npu-2.1.0.post12-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

将该目录所有文件替换到ultralytics目录下 （例：miniconda3/envs/yolo/lib/python3.10/site-packages/ultralytics）

nms默认为在cpu使用torchvision.ops.nms
要使用npu的nms 修改 331-355  ultralytics_Ascend_NPU/utils/ops.py  
npu-nms实现函数为_npu_multiclass_nms_adapter，调用的torch_npu.contrib.function.nms.npu_multiclass_nms
虽然 npu_multiclass_nms 本身很快，但其输出结果的切片操作在 NPU 上 由于 lazy 内核调度可能造成延迟，导致整体耗时偏高。因此建议在极限延迟场景下使用 CPU NMS
如果有大佬有更好的方法，欢迎交流
## 🚀 推理（Predict）

```python
from ultralytics import YOLO
import torch_npu
import cv2
import time

img_path = r'/coco8/images/val/000000000036.jpg'
img = cv2.imread(img_path)

# Initialize model
model = YOLO('yolov8n.pt').to('npu')

for i in range(100):
    start_time = time.time()
    result = model.predict(img, save=True)
    end_time = time.time()
    print("Inference time: ", end_time - start_time)
```



## 🏋️‍♂️ 训练（Train）
```python
from ultralytics import YOLO
import torch
import torch_npu

if __name__ == '__main__':
    # 检查NPU可用性
    print(f"NPU available: {torch.npu.is_available()}")
    print(f"NPU device count: {torch.npu.device_count()}")

    # Load a model
    model = YOLO('yolov8n.pt')

    # Train the model
    results = model.train(
        data="coco8.yaml", 
        epochs=1000, 
        imgsz=640,
        device='npu:0',
        amp=True  # AMP 加速
        # workers=0   # 如果多进程出问题可以设为0
    )
```