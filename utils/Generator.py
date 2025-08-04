import torch.nn as nn
import torch
from config import opt

# VN: Định nghĩa lớp ResidualBlock, một khối dư (residual block) để học các đặc trưng dư
# EN: Define ResidualBlock class, a residual block to learn residual features
class ResidualBlock(nn.Module):
    # VN: Hàm khởi tạo với số kênh đầu vào và đầu ra (mặc định 64)
    # EN: Constructor with input and output feature channels (default 64)
    def __init__(self, in_features=64, out_features=64):
        super(ResidualBlock, self).__init__()  # VN: Gọi hàm khởi tạo của lớp cha nn.Module để tích hợp với PyTorch
                                              # EN: Call the parent class nn.Module constructor for PyTorch integration
        
        # VN: Tạo khối tuần tự gồm hai lớp chập, chuẩn hóa lô và ReLU
        # EN: Create sequential block with two conv layers, batch normalization, and ReLU
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, 1, 1),  # VN: Lớp chập 3x3, giữ nguyên kích thước không gian (stride=1, padding=1)
                                                  # EN: 3x3 conv layer, preserves spatial size (stride=1, padding=1)
            nn.BatchNorm2d(in_features),  # VN: Chuẩn hóa lô để ổn định huấn luyện và cải thiện gradient
                                          # EN: Batch normalization to stabilize training and improve gradients
            nn.ReLU(inplace=True),  # VN: ReLU để thêm phi tuyến tính, inplace để tiết kiệm bộ nhớ
                                    # EN: ReLU for non-linearity, inplace to save memory
            nn.Conv2d(in_features, in_features, 3, 1, 1),  # VN: Lớp chập thứ hai, giữ nguyên số kênh và kích thước
                                                  # EN: Second conv layer, maintains channels and size
            nn.BatchNorm2d(in_features),  # VN: Chuẩn hóa lô lần thứ hai
                                          # EN: Second batch normalization
        )  # VN: Lợi ích: Khối này học các đặc trưng dư, giúp mô hình sâu hơn mà không bị mất gradient
           # EN: Benefit: This block learns residual features, enabling deeper models without vanishing gradients

    # VN: Hàm forward, thực hiện kết nối dư (residual connection)
    # EN: Forward function, implements residual connection
    def forward(self, x):
        return x + self.block(x)  # VN: Cộng đầu vào gốc với đầu ra của khối để tạo kết nối dư
                                  # EN: Add original input to block output for residual connection
        # VN: Lợi ích: Kết nối dư giúp giảm thiểu vấn đề vanishing gradient và cải thiện hiệu suất
        # EN: Benefit: Residual connection mitigates vanishing gradient issues and improves performance

# VN: Định nghĩa lớp Generator, một mạng sinh (generator) để tạo ảnh từ nhiễu và ảnh điều kiện
# EN: Define Generator class, a network to generate images from noise and conditional input
class Generator(nn.Module):
    # VN: Hàm khởi tạo cho mô hình sinh
    # EN: Constructor for the generator model
    def __init__(self):
        super(Generator, self).__init__()  # VN: Gọi hàm khởi tạo của lớp cha nn.Module
                                          # EN: Call the parent class nn.Module constructor
        
        # VN: Lớp tuyến tính để chuyển đổi vector nhiễu thành bản đồ đặc trưng dạng ảnh
        # EN: Linear layer to transform noise vector into image-shaped feature map
        self.fc = nn.Linear(opt.latent_dim, opt.channels * opt.img_size ** 2)  # VN: Ánh xạ từ latent_dim sang kênh * kích thước ảnh
                                                                      # EN: Maps from latent_dim to channels * image size
        # VN: Lợi ích: Biến đổi nhiễu thành định dạng tương thích với ảnh đầu vào
        # EN: Benefit: Transforms noise into a format compatible with input image

        # VN: Tầng đầu tiên với chập và ReLU
        # EN: First layer with convolution and ReLU
        self.l1 = nn.Sequential(
            nn.Conv2d(opt.channels * 2, 64, 3, 1, 1),  # VN: Chập 3x3, xử lý đầu vào kết hợp (ảnh + nhiễu), giảm kênh về 64
                                               # EN: 3x3 conv, processes combined input (image + noise), reduces to 64 channels
            nn.ReLU(inplace=True)  # VN: ReLU để thêm phi tuyến tính
                                   # EN: ReLU for non-linearity
        )  # VN: Lợi ích: Khởi tạo đặc trưng ban đầu từ đầu vào kết hợp
           # EN: Benefit: Initializes features from combined input

        # VN: Tạo danh sách các khối dư
        # EN: Create list of residual blocks
        resblocks = []
        for _ in range(opt.n_residual_blocks):  # VN: Tạo số lượng khối dư theo cấu hình
            resblocks.append(ResidualBlock())  # VN: Thêm từng khối ResidualBlock
                                              # EN: Add each ResidualBlock
        self.resblocks = nn.Sequential(*resblocks)  # VN: Tạo tuần tự các khối dư
                                                  # EN: Create sequential residual blocks
        # VN: Lợi ích: Nhiều khối dư giúp học đặc trưng phức tạp mà vẫn ổn định
        # EN: Benefit: Multiple residual blocks enable learning complex features while remaining stable

        # VN: Tầng cuối để tạo ảnh đầu ra
        # EN: Final layer to produce output image
        self.l2 = nn.Sequential(
            nn.Conv2d(64, opt.channels, 3, 1, 1),  # VN: Chập 3x3, chuyển về số kênh của ảnh gốc
                                           # EN: 3x3 conv, maps to original image channels
            nn.Tanh()  # VN: Tanh để chuẩn hóa đầu ra về [-1, 1]
                       # EN: Tanh to normalize output to [-1, 1]
        )  # VN: Lợi ích: Tạo ảnh đầu ra có giá trị pixel chuẩn hóa, phù hợp với dữ liệu ảnh
           # EN: Benefit: Produces normalized output image suitable for image data

    # VN: Hàm forward, tạo ảnh từ ảnh điều kiện và nhiễu
    # EN: Forward function, generates image from conditional image and noise
    def forward(self, img, z):
        # VN: Kết hợp ảnh điều kiện và nhiễu đã được chuyển đổi thành bản đồ đặc trưng
        # EN: Combine conditional image and noise transformed into feature map
        gen_input = torch.cat((img, self.fc(z).view(*img.shape)), 1)  # VN: Ánh xạ nhiễu qua fc, định dạng lại và nối với ảnh
                                                              # EN: Map noise through fc, reshape, and concatenate with image
        # VN: Lợi ích: Kết hợp thông tin từ ảnh điều kiện và nhiễu để sinh ảnh
        # EN: Benefit: Combines conditional image and noise information for generation

        out = self.l1(gen_input)  # VN: Xử lý đầu vào kết hợp qua tầng l1
                                  # EN: Process combined input through l1 layer
        out = self.resblocks(out)  # VN: Xử lý qua các khối dư để tinh chỉnh đặc trưng
                                   # EN: Process through residual blocks to refine features
        img_ = self.l2(out)  # VN: Tạo ảnh đầu ra qua tầng l2
                             # EN: Generate output image through l2 layer

        return img_  # VN: Trả về ảnh được sinh ra
                     # EN: Return generated image
        # VN: Lợi ích: Tạo ảnh chất lượng cao dựa trên điều kiện và nhiễu, tận dụng kiến trúc dư
        # EN: Benefit: Generates high-quality images from condition and noise, leveraging residual architecture