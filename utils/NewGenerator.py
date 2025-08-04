import torch.nn as nn
import torch
from utils.SelfAttention import SelfAttention
from config import opt

# VN: Định nghĩa lớp ResidualBlock, một khối dư để học các đặc trưng dư, cải thiện huấn luyện mạng sâu
# EN: Define ResidualBlock class, a residual block to learn residual features, improving deep network training
class ResidualBlock(nn.Module):
    # VN: Hàm khởi tạo với số kênh đầu vào (mặc định 64)
    # EN: Constructor with input channels (default 64)
    def __init__(self, in_channels=64):
        super(ResidualBlock, self).__init__()  # VN: Gọi hàm khởi tạo của lớp cha nn.Module để tích hợp với PyTorch
                                              # EN: Call the parent class nn.Module constructor for PyTorch integration
        
        # VN: Tạo khối tuần tự gồm hai lớp chập, chuẩn hóa lô và ReLU
        # EN: Create sequential block with two conv layers, batch normalization, and ReLU
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),  # VN: Chập 3x3, giữ kích thước không gian, không dùng bias vì có batch norm
                                                            # EN: 3x3 conv, preserves spatial size, no bias due to batch norm
            nn.BatchNorm2d(in_channels),  # VN: Chuẩn hóa lô để ổn định huấn luyện và cải thiện gradient
                                          # EN: Batch normalization to stabilize training and improve gradients
            nn.ReLU(inplace=True),  # VN: ReLU để thêm phi tuyến tính, inplace để tiết kiệm bộ nhớ
                                    # EN: ReLU for non-linearity, inplace to save memory
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),  # VN: Chập 3x3 thứ hai, giữ kích thước và không dùng bias
                                                           # EN: Second 3x3 conv, maintains size, no bias
            nn.BatchNorm2d(in_channels)  # VN: Chuẩn hóa lô lần thứ hai
                                         # EN: Second batch normalization
        )  # VN: Lợi ích: Khối này học các đặc trưng dư, giúp mô hình sâu hơn mà không bị mất gradient
           # EN: Benefit: This block learns residual features, enabling deeper models without vanishing gradients

    # VN: Hàm forward, thực hiện kết nối dư
    # EN: Forward function, implements residual connection
    def forward(self, x):
        return x + self.block(x)  # VN: Cộng đầu vào gốc với đầu ra của khối để tạo kết nối dư
                                  # EN: Add original input to block output for residual connection
        # VN: Lợi ích: Kết nối dư giảm thiểu vấn đề vanishing gradient, cải thiện hiệu suất huấn luyện
        # EN: Benefit: Residual connection mitigates vanishing gradient issues, improving training performance

# VN: Định nghĩa lớp Generator, mạng sinh để tạo ảnh từ nhiễu và ảnh điều kiện
# EN: Define Generator class, a network to generate images from noise and conditional input
class Generator(nn.Module):
    # VN: Hàm khởi tạo cho mô hình sinh
    # EN: Constructor for the generator model
    def __init__(self):
        super(Generator, self).__init__()  # VN: Gọi hàm khởi tạo của lớp cha nn.Module
                                          # EN: Call the parent class nn.Module constructor
        
        # VN: Lớp tuyến tính để chuyển đổi vector nhiễu thành bản đồ đặc trưng dạng ảnh
        # EN: Linear layer to transform noise vector into image-shaped feature map
        self.fc = nn.Linear(opt.latent_dim, opt.channels * opt.img_size ** 2)  # VN: Ánh xạ từ latent_dim sang kênh * kích thước ảnh bình phương
                                                                      # EN: Maps from latent_dim to channels * image size squared
        # VN: Lợi ích: Biến đổi nhiễu thành định dạng tương thích với ảnh đầu vào
        # EN: Benefit: Transforms noise into a format compatible with input image

        # VN: Tầng ban đầu để xử lý đầu vào kết hợp
        # EN: Initial layer to process combined input
        self.initial = nn.Sequential(
            nn.Conv2d(opt.channels * 2, 64, 3, 1, 1),  # VN: Chập 3x3, xử lý đầu vào kết hợp (ảnh + nhiễu), giảm kênh về 64
                                               # EN: 3x3 conv, processes combined input (image + noise), reduces to 64 channels
            nn.ReLU(inplace=True)  # VN: ReLU để thêm phi tuyến tính, inplace để tiết kiệm bộ nhớ
                                   # EN: ReLU for non-linearity, inplace to save memory
        )  # VN: Lợi ích: Tạo đặc trưng ban đầu từ đầu vào kết hợp, chuẩn bị cho các tầng tiếp theo
           # EN: Benefit: Generates initial features from combined input, preparing for subsequent layers

        # VN: Tạo danh sách các khối dư
        # EN: Create list of residual blocks
        resblocks = []
        for _ in range(opt.n_residual_blocks):  # VN: Tạo số lượng khối dư theo cấu hình opt.n_residual_blocks
            resblocks.append(ResidualBlock(64))  # VN: Thêm từng khối ResidualBlock với 64 kênh
                                                # EN: Add each ResidualBlock with 64 channels
        self.resblocks = nn.Sequential(*resblocks)  # VN: Tạo tuần tự các khối dư
                                                  # EN: Create sequential residual blocks
        # VN: Lợi ích: Nhiều khối dư giúp học đặc trưng phức tạp mà vẫn giữ ổn định
        # EN: Benefit: Multiple residual blocks enable learning complex features while maintaining stability

        # VN: Lớp SelfAttention để tập trung vào các vùng quan trọng trong bản đồ đặc trưng
        # EN: SelfAttention layer to focus on important regions in the feature map
        self.attn = SelfAttention(64)  # VN: Áp dụng cơ chế tự chú ý cho bản đồ đặc trưng 64 kênh
                                       # EN: Apply self-attention mechanism to 64-channel feature map
        # VN: Lợi ích: Tăng khả năng mô hình tập trung vào các mối quan hệ không gian quan trọng
        # EN: Benefit: Enhances model’s ability to focus on important spatial relationships

        # VN: Tầng cuối để tạo ảnh đầu ra
        # EN: Final layer to produce output image
        self.final = nn.Sequential(
            nn.Conv2d(64, opt.channels, 3, 1, 1),  # VN: Chập 3x3, chuyển về số kênh của ảnh gốc
                                           # EN: 3x3 conv, maps to original image channels
            nn.Tanh()  # VN: Tanh để chuẩn hóa đầu ra về [-1, 1], phù hợp với dữ liệu ảnh
                       # EN: Tanh to normalize output to [-1, 1], suitable for image data
        )  # VN: Lợi ích: Tạo ảnh đầu ra có giá trị pixel chuẩn hóa, dễ sử dụng trong pipeline
           # EN: Benefit: Produces normalized output image, easy to use in pipeline

    # VN: Hàm forward, tạo ảnh từ ảnh điều kiện và nhiễu
    # EN: Forward function, generates image from conditional image and noise
    def forward(self, img, z):
        # VN: Chuyển đổi vector nhiễu thành bản đồ đặc trưng có kích thước như ảnh
        # EN: Transform noise vector into feature map with image dimensions
        noise_projection = self.fc(z).view(img.size(0), opt.channels, opt.img_size, opt.img_size)  # VN: Ánh xạ nhiễu qua fc và định dạng lại
                                                                                         # EN: Map noise through fc and reshape
        # VN: Lợi ích: Đảm bảo nhiễu có định dạng tương thích với ảnh đầu vào
        # EN: Benefit: Ensures noise is formatted to match input image

        # VN: Kết hợp ảnh điều kiện và nhiễu đã chiếu
        # EN: Combine conditional image and projected noise
        gen_input = torch.cat((img, noise_projection), dim=1)  # VN: Nối ảnh và nhiễu theo chiều kênh
                                                              # EN: Concatenate image and noise along channel dimension
        # VN: Lợi ích: Kết hợp thông tin điều kiện và nhiễu để sinh ảnh
        # EN: Benefit: Combines conditional and noise information for generation

        out = self.initial(gen_input)  # VN: Xử lý đầu vào kết hợp qua tầng initial
                                       # EN: Process combined input through initial layer
        out = self.resblocks(out)  # VN: Xử lý qua các khối dư để tinh chỉnh đặc trưng
                                   # EN: Process through residual blocks to refine features
        out = self.attn(out)  # VN: Áp dụng tự chú ý để tăng cường đặc trưng không gian
                              # EN: Apply self-attention to enhance spatial features
        out = self.final(out)  # VN: Tạo ảnh đầu ra qua tầng final
                               # EN: Generate output image through final layer

        return out  # VN: Trả về ảnh được sinh ra
                    # EN: Return generated image
        # VN: Lợi ích: Tạo ảnh chất lượng cao với sự kết hợp của kết nối dư và tự chú ý
        # EN: Benefit: Generates high-quality images using residual connections and self-attention