import torch.nn as nn
import torch.nn.functional as F
import torch

# VN: Định nghĩa lớp SelfAttention, triển khai cơ chế tự chú ý để tập trung vào các vùng quan trọng trong bản đồ đặc trưng
# EN: Define SelfAttention class, implements self-attention mechanism to focus on important regions in feature maps
class SelfAttention(nn.Module):
    # VN: Hàm khởi tạo với số kênh đầu vào
    # EN: Constructor with input channel count
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()  # VN: Gọi hàm khởi tạo của lớp cha nn.Module để tích hợp với PyTorch
                                              # EN: Call the parent class nn.Module constructor for PyTorch integration
        
        # VN: Lớp chập 1x1 để tạo query, giảm số kênh xuống in_channels//8
        # EN: 1x1 convolution to create query, reducing channels to in_channels//8
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)  # VN: Chập 1x1 để tạo bản đồ query hiệu quả
                                                              # EN: 1x1 conv to create query map efficiently
        # VN: Lợi ích: Giảm số kênh để giảm chi phí tính toán trong cơ chế chú ý
        # EN: Benefit: Reduces channel count to lower computation cost in attention mechanism

        # VN: Lớp chập 1x1 để tạo key, cũng giảm số kênh xuống in_channels//8
        # EN: 1x1 convolution to create key, also reducing channels to in_channels//8
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)  # VN: Chập 1x1 để tạo bản đồ key
                                                            # EN: 1x1 conv to create key map
        # VN: Lợi ích: Giữ kích thước key tương thích với query để tính ma trận chú ý
        # EN: Benefit: Keeps key dimensions compatible with query for attention matrix computation

        # VN: Lớp chập 1x1 để tạo value, giữ nguyên số kênh
        # EN: 1x1 convolution to create value, maintaining original channel count
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)  # VN: Chập 1x1 để tạo bản đồ value
                                                         # EN: 1x1 conv to create value map
        # VN: Lợi ích: Duy trì đầy đủ thông tin đặc trưng trong value
        # EN: Benefit: Preserves full feature information in value

        # VN: Tham số học được gamma để điều chỉnh mức độ ảnh hưởng của chú ý
        # EN: Learnable parameter gamma to scale the attention output
        self.gamma = nn.Parameter(torch.zeros(1))  # VN: Khởi tạo gamma bằng 0, sẽ được học trong quá trình huấn luyện
                                                  # EN: Initialize gamma as 0, to be learned during training
        # VN: Lợi ích: Cho phép mô hình tự điều chỉnh tầm quan trọng của cơ chế chú ý
        # EN: Benefit: Allows the model to learn the importance of the attention mechanism

    # VN: Hàm forward, thực hiện cơ chế tự chú ý và kết nối dư
    # EN: Forward function, implements self-attention and residual connection
    def forward(self, x):
        batch_size, C, H, W = x.size()  # VN: Lấy kích thước đầu vào: batch_size, số kênh (C), chiều cao (H), chiều rộng (W)
                                        # EN: Get input dimensions: batch_size, channels (C), height (H), width (W)
        
        # VN: Tạo query: B x (H*W) x (C//8)
        # EN: Create query: B x (H*W) x (C//8)
        proj_query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # VN: Chập query, làm phẳng thành (B, C//8, H*W), rồi chuyển vị thành (B, H*W, C//8)
                                                                              # EN: Apply query conv, flatten to (B, C//8, H*W), then permute to (B, H*W, C//8)
        # VN: Lợi ích: Chuẩn bị query cho phép tính toán ma trận chú ý hiệu quả
        # EN: Benefit: Prepares query for efficient attention matrix computation

        # VN: Tạo key: B x (C//8) x (H*W)
        # EN: Create key: B x (C//8) x (H*W)
        proj_key = self.key_conv(x).view(batch_size, -1, H * W)  # VN: Chập key, làm phẳng thành (B, C//8, H*W)
                                                                # EN: Apply key conv, flatten to (B, C//8, H*W)
        # VN: Lợi ích: Đảm bảo key tương thích với query để tính tương quan không gian
        # EN: Benefit: Ensures key is compatible with query for spatial correlation

        # VN: Tính ma trận chú ý: B x (H*W) x (H*W)
        # EN: Compute attention map: B x (H*W) x (H*W)
        energy = torch.bmm(proj_query, proj_key)  # VN: Nhân batch matrix giữa query và key để tính tương quan
                                                  # EN: Batch matrix multiplication between query and key for correlation
        attention = F.softmax(energy, dim=-1)  # VN: Áp dụng softmax để chuẩn hóa thành phân phối chú ý
                                               # EN: Apply softmax to normalize into attention distribution
        # VN: Lợi ích: Ma trận chú ý xác định mức độ quan trọng của từng vị trí không gian
        # EN: Benefit: Attention map determines the importance of each spatial position

        # VN: Tạo value: B x C x (H*W)
        # EN: Create value: B x C x (H*W)
        proj_value = self.value_conv(x).view(batch_size, -1, H * W)  # VN: Chập value, làm phẳng thành (B, C, H*W)
                                                                    # EN: Apply value conv, flatten to (B, C, H*W)
        # VN: Lợi ích: Giữ toàn bộ thông tin đặc trưng để áp dụng chú ý
        # EN: Benefit: Preserves full feature information for attention application

        # VN: Áp dụng chú ý: B x C x (H*W)
        # EN: Apply attention: B x C x (H*W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # VN: Nhân value với ma trận chú ý (chuyển vị) để tái trọng số đặc trưng
                                                                # EN: Multiply value with transposed attention map to reweight features
        out = out.view(batch_size, C, H, W)  # VN: Định dạng lại đầu ra thành kích thước gốc (B, C, H, W)
                                             # EN: Reshape output to original dimensions (B, C, H, W)
        # VN: Lợi ích: Tạo bản đồ đặc trưng tái trọng số dựa trên mối quan hệ không gian
        # EN: Benefit: Produces reweighted feature map based on spatial relationships

        # VN: Kết nối dư với trọng số học được
        # EN: Residual connection with learnable weight
        out = self.gamma * out + x  # VN: Cộng đầu ra chú ý (được nhân với gamma) với đầu vào gốc
                                    # EN: Add scaled attention output (via gamma) to original input
        # VN: Lợi ích: Kết nối dư đảm bảo mô hình giữ được đặc trưng gốc và ổn định huấn luyện
        # EN: Benefit: Residual connection ensures retention of original features and stabilizes training

        return out  # VN: Trả về bản đồ đặc trưng sau khi áp dụng chú ý
                    # EN: Return feature map after applying attention
        # VN: Lợi ích: Tăng cường khả năng mô hình tập trung vào các vùng quan trọng, cải thiện hiệu suất
        # EN: Benefit: Enhances model’s focus on important regions, improving performance