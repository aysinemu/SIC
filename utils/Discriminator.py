import torch.nn as nn
from config import opt

# VN: Định nghĩa lớp Discriminator, một mạng nơ-ron phân biệt dùng trong huấn luyện đối kháng
# EN: Define Discriminator class, a neural network for adversarial training to distinguish real vs. fake images
class Discriminator(nn.Module):
    # VN: Hàm khởi tạo cho mô hình phân biệt
    # EN: Constructor for the discriminator model
    def __init__(self):
        super(Discriminator, self).__init__()  # VN: Gọi hàm khởi tạo của lớp cha nn.Module để tích hợp với PyTorch
                                              # EN: Call the parent class nn.Module constructor for PyTorch integration
        
        # VN: Hàm block tạo một khối mạng với Conv2d, LeakyReLU và tùy chọn chuẩn hóa
        # EN: block function creates a network block with Conv2d, LeakyReLU, and optional normalization
        def block(in_features, out_features, normalization=True):
            """Discriminator block"""
            # VN: Tạo danh sách các tầng: chập 3x3, LeakyReLU và chuẩn hóa (nếu bật)
            # EN: Create list of layers: 3x3 convolution, LeakyReLU, and normalization (if enabled)
            layers = [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),  # VN: Lớp chập giảm kích thước không gian (stride=2), tăng số kênh
                                                                      # EN: Conv layer reduces spatial size (stride=2), increases channels
                nn.LeakyReLU(0.2, inplace=True)  # VN: LeakyReLU với độ dốc 0.2 để tránh chết nơ-ron và ổn định huấn luyện
                                                 # EN: LeakyReLU with 0.2 slope to avoid dead neurons and stabilize training
            ]
            if normalization:  # VN: Nếu bật chuẩn hóa, thêm InstanceNorm2d
                layers.append(nn.InstanceNorm2d(out_features))  # VN: Chuẩn hóa instance để giảm sự khác biệt về phong cách giữa các ảnh
                                                              # EN: Instance normalization to reduce style differences across images
            # VN: Lợi ích: Khối này cung cấp mô-đun tái sử dụng để xây dựng mạng phân biệt sâu
            # EN: Benefit: This block provides a reusable module for building a deep discriminator
            return layers

        # VN: Xây dựng mô hình chính với các khối liên tiếp và tầng chập cuối
        # EN: Build the main model with sequential blocks and a final conv layer
        self.model = nn.Sequential(
            *block(opt.channels, 64, normalization=False),  # VN: Khối đầu tiên, không chuẩn hóa, chuyển từ số kênh đầu vào sang 64
                                                          # EN: First block, no normalization, maps from input channels to 64
            *block(64, 128),  # VN: Khối thứ hai, tăng kênh lên 128
                              # EN: Second block, increases channels to 128
            *block(128, 256),  # VN: Khối thứ ba, tăng kênh lên 256
                               # EN: Third block, increases channels to 256
            *block(256, 512),  # VN: Khối thứ tư, tăng kênh lên 512
                               # EN: Fourth block, increases channels to 512
            nn.Conv2d(512, 1, 3, 1, 1)  # VN: Lớp chập cuối, tạo đầu ra 1 kênh (điểm số phân biệt)
                                        # EN: Final conv layer, produces 1-channel output (discrimination score)
        )  # VN: Lợi ích: Kiến trúc phân cấp giúp trích xuất đặc trưng phức tạp để đánh giá tính chân thực của ảnh
           # EN: Benefit: Hierarchical architecture extracts complex features to assess image authenticity

    # VN: Hàm forward, xử lý đầu vào và trả về điểm số phân biệt
    # EN: Forward function, processes input and returns discrimination score
    def forward(self, img):
        validity = self.model(img)  # VN: Xử lý ảnh qua các khối chập để tạo bản đồ điểm số phân biệt
                                   # EN: Process image through conv blocks to generate discrimination score map
        # VN: Lợi ích: Đầu ra là bản đồ điểm số, cho phép phân biệt chi tiết giữa ảnh thật và giả
        # EN: Benefit: Output is a score map, enabling detailed discrimination between real and fake images
        return validity  # VN: Trả về bản đồ điểm số phân biệt
                         # EN: Return discrimination score map
        # VN: Lợi ích: Cung cấp đầu ra linh hoạt, có thể được sử dụng trong hàm mất mát đối kháng
        # EN: Benefit: Provides flexible output, usable in adversarial loss functions