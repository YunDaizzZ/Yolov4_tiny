# coding:utf-8
import torch
from torchsummary import summary
from nets.yolov4_tiny import YoloBody

if __name__ == "__main__":
    # 需要使用device来指定网络在GPU还是CPU运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = YoloBody(3, 20).to(device)
    summary(m, input_size=(3, 416, 416))