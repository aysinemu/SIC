import math  
import torch.nn as nn  
from utils.SelfAttention import SelfAttention  
from config import opt

# VN: Định nghĩa lớp DomainConfusionDiscriminator, một mạng phân biệt để phân loại miền trong GAN
# EN: Define DomainConfusionDiscriminator class, a discriminator network to classify domains in a GAN
class DomainConfusionDiscriminator(nn.Module):
    # VN: Hàm khởi tạo cho mạng phân biệt
    # EN: Constructor for the discriminator network
    def __init__(self):
        super().__init__()  # VN: Gọi hàm khởi tạo của lớp cha nn.Module để tích hợp với PyTorch
                            # EN: Call the parent class nn.Module constructor for PyTorch integration
        
        # VN: Tạo chuỗi các tầng xử lý tuần tự
        # EN: Create a sequential stack of processing layers
        self.layers = nn.Sequential(
            # VN: Tầng chập đầu tiên: từ số kênh đầu vào (opt.channels) sang 64 kênh
            # EN: First convolutional layer: from input channels (opt.channels) to 64 channels
            nn.Conv2d(opt.channels, 64, 4, 2, 1),  # VN: Chập 4x4, stride=2, padding=1, giảm kích thước không gian
                                           # EN: 4x4 conv, stride=2, padding=1, reduces spatial size
            nn.LeakyReLU(0.2, inplace=True),  # VN: LeakyReLU với độ dốc 0.2, thêm phi tuyến tính
                                              # EN: LeakyReLU with slope 0.2, adds non-linearity
            # VN: Lợi ích: Trích xuất đặc trưng cấp thấp, giảm kích thước không gian
            # EN: Benefit: Extracts low-level features, reduces spatial dimensions

            # VN: Tầng chập thứ hai: từ 64 kênh sang 128 kênh
            # EN: Second convolutional layer: from 64 channels to 128 channels
            nn.Conv2d(64, 128, 4, 2, 1),  # VN: Chập 4x4, stride=2, tiếp tục giảm kích thước
                                           # EN: 4x4 conv, stride=2, further reduces spatial size
            nn.InstanceNorm2d(128),  # VN: Chuẩn hóa instance để ổn định huấn luyện
                                     # EN: Instance normalization to stabilize training
            nn.LeakyReLU(0.2, inplace=True),  # VN: LeakyReLU thứ hai
                                              # EN: Second LeakyReLU
            # VN: Lợi ích: Trích xuất đặc trưng trung gian, chuẩn hóa giúp cải thiện gradient
            # EN: Benefit: Extracts mid-level features, normalization improves gradients

            # VN: Tầng chập thứ ba: từ 128 kênh sang 256 kênh
            # EN: Third convolutional layer: from 128 channels to 256 channels
            nn.Conv2d(128, 256, 4, 2, 1),  # VN: Chập 4x4, stride=2, giảm kích thước hơn nữa
                                           # EN: 4x4 conv, stride=2, further reduces spatial size
            nn.InstanceNorm2d(256),  # VN: Chuẩn hóa instance cho 256 kênh
                                     # EN: Instance normalization for 256 channels
            nn.LeakyReLU(0.2, inplace=True),  # VN: LeakyReLU thứ ba
                                              # EN: Third LeakyReLU
            # VN: Lợi ích: Tăng số kênh để trích xuất đặc trưng phức tạp hơn
            # EN: Benefit: Increases channels to extract more complex features

            # VN: Tầng tự chú ý để tập trung vào các vùng quan trọng
            # EN: Self-attention layer to focus on important regions
            SelfAttention(256),  # VN: Áp dụng cơ chế tự chú ý cho bản đồ đặc trưng 256 kênh
                                 # EN: Apply self-attention mechanism to 256-channel feature map
            # VN: Lợi ích: Giúp mô hình tập trung vào các mối quan hệ không gian quan trọng
            # EN: Benefit: Enables model to focus on important spatial relationships

            # VN: Tầng chập thứ tư: từ 256 kênh sang 512 kênh
            # EN: Fourth convolutional layer: from 256 channels to 512 channels
            nn.Conv2d(256, 512, 4, 2, 1),  # VN: Chập 4x4, stride=2, giảm kích thước lần cuối
                                           # EN: 4x4 conv, stride=2, final spatial size reduction
            nn.InstanceNorm2d(512),  # VN: Chuẩn hóa instance cho 512 kênh
                                     # EN: Instance normalization for 512 channels
            nn.LeakyReLU(0.2, inplace=True),  # VN: LeakyReLU thứ tư
                                              # EN: Fourth LeakyReLU
            # VN: Lợi ích: Trích xuất đặc trưng cấp cao với số lượng kênh lớn
            # EN: Benefit: Extracts high-level features with large channel count

            # VN: Tầng pooling để giảm kích thước đầu ra xuống 1x1
            # EN: Pooling layer to reduce output size to 1x1
            nn.AdaptiveAvgPool2d(1),  # VN: Pooling thích ứng để tạo đầu ra 1x1 bất kể kích thước đầu vào
                                      # EN: Adaptive average pooling to produce 1x1 output regardless of input size
            # VN: Lợi ích: Đảm bảo đầu ra có kích thước cố định để dễ dàng phân loại
            # EN: Benefit: Ensures fixed-size output for easy classification

            # VN: Tầng chập cuối để ánh xạ sang đầu ra 1 kênh
            # EN: Final convolutional layer to map to 1-channel output
            nn.Conv2d(512, 1, 1)  # VN: Chập 1x1 để tạo đầu ra phân biệt (thật/giả)
                                  # EN: 1x1 conv to produce discriminator output (real/fake)
            # VN: Lợi ích: Tạo đầu ra phân biệt đơn giản để sử dụng trong mất mát đối kháng
            # EN: Benefit: Produces simple discriminator output for adversarial loss
        )  # VN: Lợi ích: Kiến trúc tuần tự giúp đơn giản hóa xử lý và trích xuất đặc trưng phân biệt
           # EN: Benefit: Sequential architecture simplifies processing and feature extraction for discrimination

    # VN: Hàm forward, xử lý ảnh đầu vào và trả về đầu ra phân biệt
    # EN: Forward function, processes input image and returns discriminator output
    def forward(self, x):
        return self.layers(x).view(x.size(0), -1)  # VN: Xử lý qua các tầng và định dạng lại đầu ra thành (batch_size, 1)
                                                  # EN: Process through layers and reshape output to (batch_size, 1)
        # VN: Lợi ích: Đầu ra phù hợp để tính mất mát đối kháng, phân biệt ảnh thật và giả
        # EN: Benefit: Output is suitable for adversarial loss, distinguishing real and fake images
