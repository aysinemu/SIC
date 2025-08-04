import torch  
import math  
import torch.nn as nn  
import torch.nn.functional as F 
from utils.SelfAttention import SelfAttention  
from config import opt

# VN: Định nghĩa lớp UNetBlock, một khối cơ bản trong kiến trúc U-Net
# EN: Define UNetBlock class, a basic building block in the U-Net architecture
class UNetBlock(nn.Module):
    # VN: Hàm khởi tạo với số kênh vào, ra và chiều thời gian
    # EN: Constructor with input channels, output channels, and time dimension
    def __init__(self, in_channels, out_channels, time_dim=128):
        super().__init__()  # VN: Gọi hàm khởi tạo của lớp cha nn.Module
                            # EN: Call the parent class nn.Module constructor
        # VN: MLP để xử lý embedding thời gian, chuyển từ time_dim sang out_channels
        # EN: MLP to process time embedding, mapping from time_dim to out_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, out_channels),  # VN: Lớp tuyến tính để biến đổi embedding thời gian
                                               # EN: Linear layer to transform time embedding
            nn.ReLU()  # VN: Hàm kích hoạt ReLU để thêm phi tuyến tính
                       # EN: ReLU activation to add non-linearity
        )  # VN: Lợi ích: Cho phép điều chỉnh embedding thời gian phù hợp với kênh đầu ra
           # EN: Benefit: Allows time embedding to be adjusted to match output channels
        # VN: Khối chập với hai lớp Conv2d, chuẩn hóa lô và ReLU
        # EN: Convolutional block with two Conv2d layers, batch norm, and ReLU
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),  # VN: Lớp chập 3x3, giữ nguyên kích thước không gian
                                                        # EN: 3x3 convolution, preserving spatial dimensions
            nn.BatchNorm2d(out_channels),  # VN: Chuẩn hóa lô để ổn định huấn luyện
                                          # EN: Batch normalization to stabilize training
            nn.ReLU(),  # VN: ReLU để thêm phi tuyến tính
                        # EN: ReLU to add non-linearity
            nn.Conv2d(out_channels, out_channels, 3, padding=1),  # VN: Lớp chập thứ hai để tăng khả năng học đặc trưng
                                                         # EN: Second convolution to enhance feature learning
            nn.BatchNorm2d(out_channels),  # VN: Chuẩn hóa lô lần hai
                                          # EN: Second batch normalization
            nn.ReLU()  # VN: ReLU lần hai
                       # EN: Second ReLU
        )  # VN: Lợi ích: Khối chập này trích xuất đặc trưng không gian mạnh mẽ và ổn định
           # EN: Benefit: This conv block extracts robust spatial features and stabilizes training

    # VN: Hàm forward, xử lý đầu vào x và embedding thời gian t
    # EN: Forward function, processes input x and time embedding t
    def forward(self, x, t):
        t_emb = self.time_mlp(t)  # VN: Chuyển đổi embedding thời gian qua MLP
                                  # EN: Transform time embedding through MLP
        t_emb = t_emb[(...,) + (None,) * 2]  # VN: Định dạng lại t_emb để phù hợp với kích thước đặc trưng (batch, out_channels, 1, 1)
                                             # EN: Reshape t_emb to match feature map shape (batch, out_channels, 1, 1)
        # VN: Lợi ích: Cho phép cộng embedding thời gian vào bản đồ đặc trưng
        # EN: Benefit: Enables adding time embedding to feature maps
        x = self.conv(x) + t_emb  # VN: Áp dụng khối chập và cộng embedding thời gian
                                  # EN: Apply conv block and add time embedding
        # VN: Lợi ích: Kết hợp thông tin không gian và thời gian
        # EN: Benefit: Combines spatial and temporal information
        return x  # VN: Trả về bản đồ đặc trưng đã xử lý
                  # EN: Return the processed feature map

# VN: Định nghĩa lớp DiffusionModel, triển khai kiến trúc U-Net cho mô hình khuếch tán
# EN: Define DiffusionModel class, implementing U-Net architecture for diffusion model
class DiffusionModel(nn.Module):
    # VN: Khởi tạo với số kênh đầu vào, kích thước ảnh và chiều thời gian
    # EN: Initialize with input channels, image size, and time dimension
    def __init__(self, channels=3, img_size=opt.img_size, time_dim=128):
        super().__init__()  # VN: Gọi hàm khởi tạo của lớp cha
                            # EN: Call parent class constructor
        # VN: Khối U-Net cho đường xuống (downsampling)
        # EN: U-Net blocks for downsampling path
        self.down1 = UNetBlock(channels, 64, time_dim)  # VN: Khối đầu tiên, chuyển từ channels sang 64 kênh
                                                        # EN: First block, maps from channels to 64 channels
        self.down2 = UNetBlock(64, 128, time_dim)  # VN: Khối thứ hai, chuyển sang 128 kênh
                                                  # EN: Second block, maps to 128 channels
        self.pool = nn.MaxPool2d(2)  # VN: Lớp max pooling giảm kích thước không gian
                                     # EN: Max pooling layer to reduce spatial dimensions
        # VN: Lợi ích: Giảm kích thước để học các đặc trưng cấp cao hơn
        # EN: Benefit: Reduces size to learn higher-level features
        
        # VN: Khối bottleneck để xử lý đặc trưng cấp cao
        # EN: Bottleneck block for high-level feature processing
        self.bottleneck = UNetBlock(128, 256, time_dim)  # VN: Chuyển sang 256 kênh ở điểm nghẽn
                                                        # EN: Maps to 256 channels at bottleneck
        
        # VN: Đường lên (upsampling) với các lớp ConvTranspose và UNetBlock
        # EN: Upsampling path with ConvTranspose and UNetBlock
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)  # VN: Lớp chuyển vị chập để tăng kích thước
                                                             # EN: Transposed conv to upsample
        self.up2 = nn.ConvTranspose2d(128 + 128, 64, 2, stride=2)  # VN: Kết hợp skip connection (128+128) và upsample
                                                                  # EN: Combines skip connection (128+128) and upsamples
        self.up3 = UNetBlock(64 + 64, 64, time_dim)  # VN: Khối U-Net xử lý đặc trưng kết hợp
                                                    # EN: U-Net block for combined features
        self.out = nn.Conv2d(64, opt.channels, 3, padding=1)  # VN: Lớp chập cuối để tạo đầu ra (số kênh ban đầu)
                                                             # EN: Final conv to produce output (original channels)
        # VN: Lợi ích: Skip connections giữ lại chi tiết không gian từ đường xuống
        # EN: Benefit: Skip connections preserve spatial details from downsampling path
        
        # VN: Tạo embedding vị trí thời gian cố định
        # EN: Create fixed time positional embeddings
        self.time_pos_emb = nn.Parameter(self.positional_encoding(opt.step, time_dim), requires_grad=False)
        # VN: Lợi ích: Cung cấp biểu diễn thời gian ổn định, không cần học
        # EN: Benefit: Provides stable, non-trainable time representation

    # VN: Hàm tạo mã hóa vị trí thời gian (sinusoidal positional encoding)
    # EN: Function to create sinusoidal positional encoding for timesteps
    def positional_encoding(self, timesteps, dim):
        half_dim = dim // 2  # VN: Tính nửa chiều của embedding
                             # EN: Compute half the embedding dimension
        emb = math.log(10000) / (half_dim - 1)  # VN: Tính hệ số tỷ lệ tần số
                                                # EN: Compute frequency scaling factor
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)  # VN: Tạo tần số giảm dần
                                                                          # EN: Create exponentially decreasing frequencies
        emb = torch.arange(timesteps, dtype=torch.float)[:, None] * emb[None, :]  # VN: Nhân thời gian với tần số
                                                                                 # EN: Multiply timesteps with frequencies
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # VN: Kết hợp sin và cos để tạo embedding
                                                                  # EN: Concatenate sin and cos for embedding
        # VN: Lợi ích: Tạo biểu diễn thời gian phong phú, phân biệt các bước thời gian
        # EN: Benefit: Creates rich time representation, distinguishing timesteps
        return emb

    # VN: Hàm forward, xử lý đầu vào x và bước thời gian t
    # EN: Forward function, processes input x and timestep t
    def forward(self, x, t):
        t_emb = self.time_pos_emb[t].to(x.device)  # VN: Lấy embedding thời gian và chuyển sang thiết bị của x
                                                  # EN: Get time embedding and move to input device
        
        d1 = self.down1(x, t_emb)  # VN: Xử lý qua khối down1
                                   # EN: Process through down1 block
        d2 = self.down2(self.pool(d1), t_emb)  # VN: Pooling và xử lý qua down2
                                              # EN: Pool and process through down2
        
        b = self.bottleneck(self.pool(d2), t_emb)  # VN: Xử lý qua bottleneck
                                                  # EN: Process through bottleneck
        
        u1 = self.up1(b)  # VN: Upsample qua up1
                         # EN: Upsample through up1
        u2 = self.up2(torch.cat([u1, d2], dim=1))  # VN: Kết hợp skip connection với d2 và upsample
                                                  # EN: Combine skip connection with d2 and upsample
        u3 = self.up3(torch.cat([u2, d1], dim=1), t_emb)  # VN: Kết hợp skip connection với d1 và xử lý
                                                         # EN: Combine skip connection with d1 and process
        
        return self.out(u3)  # VN: Tạo đầu ra cuối cùng
                            # EN: Produce final output

# VN: Định nghĩa lớp ConditionalDiffusionModel, hỗ trợ khuếch tán có điều kiện
# EN: Define ConditionalDiffusionModel class, supporting conditional diffusion
class ConditionalDiffusionModel(nn.Module):
    # VN: Khởi tạo với số kênh, kích thước ảnh và chiều thời gian
    # EN: Initialize with channels, image size, and time dimension
    def __init__(self, channels=opt.channels, img_size=opt.img_size, time_dim=128):
        super().__init__()  # VN: Gọi hàm khởi tạo của lớp cha
                            # EN: Call parent class constructor
        # VN: Bộ mã hóa để xử lý điều kiện (condition)
        # EN: Encoder to process the condition
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1),  # VN: Lớp chập để trích xuất đặc trưng từ điều kiện
                                                   # EN: Conv layer to extract features from condition
            nn.ReLU(),  # VN: ReLU để thêm phi tuyến tính
                        # EN: ReLU to add non-linearity
            nn.Conv2d(64, 64, 3, padding=1),  # VN: Lớp chập thứ hai để tinh chỉnh đặc trưng
                                             # EN: Second conv to refine features
            nn.ReLU()  # VN: ReLU thứ hai
                       # EN: Second ReLU
        )  # VN: Lợi ích: Trích xuất đặc trưng điều kiện mạnh mẽ
           # EN: Benefit: Extracts robust condition features
        
        # VN: Khởi tạo U-Net với số kênh đầu vào tăng (kết hợp với đặc trưng điều kiện)
        # EN: Initialize U-Net with increased input channels (combined with condition features)
        self.unet = DiffusionModel(channels=channels + 64, img_size=img_size, time_dim=time_dim)
        # VN: Lợi ích: Kết hợp đặc trưng điều kiện với U-Net
        # EN: Benefit: Integrates condition features with U-Net

    # VN: Hàm forward, xử lý đầu vào x, bước thời gian t và điều kiện
    # EN: Forward function, processes input x, timestep t, and condition
    def forward(self, x, t, condition):
        cond_features = self.encoder(condition)  # VN: Trích xuất đặc trưng từ điều kiện
                                                # EN: Extract features from condition
        x = torch.cat([x, cond_features], dim=1)  # VN: Kết hợp đặc trưng điều kiện với đầu vào
                                                 # EN: Concatenate condition features with input
        # VN: Lợi ích: Cho phép mô hình sử dụng thông tin điều kiện để tạo ảnh
        # EN: Benefit: Allows model to use condition for image generation
        return self.unet(x, t)  # VN: Chuyển qua U-Net để xử lý tiếp
                                # EN: Pass through U-Net for further processing

# VN: Định nghĩa lớp DiffusionStyleAnchor, dùng để trích xuất đặc trưng phong cách từ mô hình khuếch tán
# EN: Define DiffusionStyleAnchor class, used to extract style features from the diffusion model
class DiffusionStyleAnchor(nn.Module):
    # VN: Hàm khởi tạo với mô hình khuếch tán được cung cấp
    # EN: Constructor with provided diffusion model
    def __init__(self, diffusion_model):
        super().__init__()  # VN: Gọi hàm khởi tạo của lớp cha nn.Module để tích hợp với PyTorch
                            # EN: Call the parent class nn.Module constructor for PyTorch integration
        self.diffusion_model = diffusion_model  # VN: Lưu tham chiếu đến mô hình khuếch tán
                                               # EN: Store reference to the diffusion model
        # VN: Tạo encoder để trích xuất đặc trưng từ ảnh miền mục tiêu
        # EN: Create encoder to extract features from target domain images
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  # VN: Lớp chập 3x3, chuyển từ 3 kênh (RGB) sang 64 kênh, giữ kích thước không gian
                                      # EN: 3x3 conv layer, maps from 3 channels (RGB) to 64 channels, preserves spatial size
            nn.ReLU(),  # VN: ReLU để thêm phi tuyến tính
                        # EN: ReLU for non-linearity
            nn.Conv2d(64, 64, 3, padding=1),  # VN: Lớp chập 3x3 thứ hai, giữ 64 kênh và kích thước không gian
                                       # EN: Second 3x3 conv layer, maintains 64 channels and spatial size
            nn.ReLU()  # VN: ReLU thứ hai để tăng phi tuyến tính
                       # EN: Second ReLU to enhance non-linearity
        )  # VN: Lợi ích: Encoder trích xuất đặc trưng cấp cao từ ảnh mục tiêu, tăng cường thông tin điều kiện
           # EN: Benefit: Encoder extracts high-level features from target images, enhancing conditional information

    # VN: Hàm forward, trích xuất đặc trưng phong cách từ các tầng của mô hình khuếch tán
    # EN: Forward function, extracts style features from layers of the diffusion model
    def forward(self, x):
        # VN: Tạo tensor timestep t bằng 0 cho toàn bộ batch
        # EN: Create timestep tensor t as zeros for the entire batch
        t = torch.zeros(x.size(0), device=x.device, dtype=torch.long)  # VN: Đặt t=0 để sử dụng bước khuếch tán đầu tiên
                                                              # EN: Set t=0 to use the first diffusion step
        # VN: Lấy embedding thời gian từ U-Net của mô hình khuếch tán
        # EN: Get time embedding from the U-Net of the diffusion model
        t_emb = self.diffusion_model.unet.time_pos_emb[t].to(x.device)  # VN: Truy xuất và chuyển embedding thời gian sang thiết bị
                                                               # EN: Retrieve and move time embedding to device
        # VN: Lợi ích: Embedding thời gian cung cấp ngữ cảnh khuếch tán cho mô hình
        # EN: Benefit: Time embedding provides diffusion context for the model

        # VN: Trích xuất đặc trưng điều kiện từ ảnh đầu vào
        # EN: Extract condition features from input image
        cond_features = self.encoder(x)  # VN: Áp dụng encoder để tạo bản đồ đặc trưng 64 kênh
                                         # EN: Apply encoder to create 64-channel feature map
        # VN: Kết hợp ảnh gốc với đặc trưng điều kiện
        # EN: Combine original image with condition features
        x = torch.cat([x, cond_features], dim=1)  # VN: Nối ảnh gốc (3 kênh) và đặc trưng (64 kênh) theo chiều kênh
                                                  # EN: Concatenate original image (3 channels) and features (64 channels) along channel dimension
        # VN: Lợi ích: Tăng cường thông tin đầu vào với đặc trưng điều kiện để cải thiện chất lượng đặc trưng
        # EN: Benefit: Enhances input with conditional features to improve feature quality

        # VN: Danh sách lưu đặc trưng phong cách từ các tầng
        # EN: List to store style features from different layers
        style_features = []

        # VN: Trích xuất đặc trưng từ tầng down1 của U-Net
        # EN: Extract features from down1 layer of U-Net
        d1 = self.diffusion_model.unet.down1(x, t_emb)  # VN: Xử lý đầu vào qua tầng down1 với embedding thời gian
                                                       # EN: Process input through down1 layer with time embedding
        style_features.append(d1)  # VN: Lưu đặc trưng từ down1
                                   # EN: Store features from down1
        # VN: Lợi ích: Đặc trưng từ down1 chứa thông tin cấp thấp của ảnh
        # EN: Benefit: Features from down1 contain low-level image information

        # VN: Trích xuất đặc trưng từ tầng down2 của U-Net
        # EN: Extract features from down2 layer of U-Net
        d2 = self.diffusion_model.unet.down2(self.diffusion_model.unet.pool(d1), t_emb)  # VN: Pooling d1 và xử lý qua down2
                                                                                # EN: Pool d1 and process through down2
        style_features.append(d2)  # VN: Lưu đặc trưng từ down2
                                   # EN: Store features from down2
        # VN: Lợi ích: Đặc trưng từ down2 chứa thông tin trung gian, giảm kích thước không gian
        # EN: Benefit: Features from down2 contain mid-level information with reduced spatial size

        # VN: Trích xuất đặc trưng từ tầng bottleneck của U-Net
        # EN: Extract features from bottleneck layer of U-Net
        b = self.diffusion_model.unet.bottleneck(self.diffusion_model.unet.pool(d2), t_emb)  # VN: Pooling d2 và xử lý qua bottleneck
                                                                                    # EN: Pool d2 and process through bottleneck
        style_features.append(b)  # VN: Lưu đặc trưng từ bottleneck
                                  # EN: Store features from bottleneck
        # VN: Lợi ích: Đặc trưng từ bottleneck chứa thông tin ngữ nghĩa cấp cao
        # EN: Benefit: Features from bottleneck contain high-level semantic information

        # VN: Trích xuất đặc trưng từ tầng up1 của U-Net
        # EN: Extract features from up1 layer of U-Net
        u1 = self.diffusion_model.unet.up1(b)  # VN: Xử lý bottleneck qua up1
                                               # EN: Process bottleneck through up1
        style_features.append(u1)  # VN: Lưu đặc trưng từ up1
                                   # EN: Store features from up1
        # VN: Lợi ích: Đặc trưng từ up1 bắt đầu tái tạo thông tin không gian
        # EN: Benefit: Features from up1 begin reconstructing spatial information

        # VN: Trích xuất đặc trưng từ tầng up2 của U-Net
        # EN: Extract features from up2 layer of U-Net
        u2 = self.diffusion_model.unet.up2(torch.cat([u1, d2], dim=1))  # VN: Kết hợp u1 và d2, xử lý qua up2
                                                               # EN: Concatenate u1 and d2, process through up2
        style_features.append(u2)  # VN: Lưu đặc trưng từ up2
                                   # EN: Store features from up2
        # VN: Lợi ích: Đặc trưng từ up2 tích hợp thông tin từ tầng trước để tái tạo chi tiết
        # EN: Benefit: Features from up2 integrate prior layer information to reconstruct details

        return style_features  # VN: Trả về danh sách các đặc trưng phong cách
                               # EN: Return list of style features
        # VN: Lợi ích: Cung cấp đặc trưng phong cách từ nhiều tầng để hướng dẫn sinh ảnh
        # EN: Benefit: Provides style features from multiple layers to guide image generation

# VN: Định nghĩa lớp SemanticConsistencyModule, dùng để đảm bảo nhất quán ngữ nghĩa giữa ảnh nguồn và ảnh sinh
# EN: Define SemanticConsistencyModule class, used to ensure semantic consistency between source and generated images
class SemanticConsistencyModule(nn.Module):
    # VN: Hàm khởi tạo với classifier được cung cấp
    # EN: Constructor with provided classifier
    def __init__(self, classifier):
        super().__init__()  # VN: Gọi hàm khởi tạo của lớp cha nn.Module để tích hợp với PyTorch
                            # EN: Call the parent class nn.Module constructor for PyTorch integration
        self.classifier = classifier  # VN: Lưu tham chiếu đến classifier để trích xuất đặc trưng
                                      # EN: Store reference to classifier for feature extraction
        # VN: Lợi ích: Tận dụng classifier đã huấn luyện để trích xuất đặc trưng ngữ nghĩa
        # EN: Benefit: Leverages pre-trained classifier to extract semantic features

    # VN: Hàm extract_features, trích xuất đặc trưng trung gian từ classifier
    # EN: extract_features function, extracts intermediate features from classifier
    def extract_features(self, x):
        x = self.classifier.conv1(x)  # VN: Áp dụng tầng chập đầu tiên của classifier
                                      # EN: Apply classifier's first convolutional layer
        x = self.classifier.relu1(x)  # VN: Áp dụng ReLU để thêm phi tuyến tính
                                      # EN: Apply ReLU for non-linearity
        x = self.classifier.pool1(x)  # VN: Áp dụng max pooling để giảm kích thước không gian
                                      # EN: Apply max pooling to reduce spatial dimensions
        x = self.classifier.conv2(x)  # VN: Áp dụng tầng chập thứ hai
                                      # EN: Apply second convolutional layer
        x = self.classifier.relu2(x)  # VN: Áp dụng ReLU thứ hai
                                      # EN: Apply second ReLU
        x = self.classifier.pool2(x)  # VN: Áp dụng max pooling thứ hai
                                      # EN: Apply second max pooling
        return x  # VN: Trả về bản đồ đặc trưng trung gian
                  # EN: Return intermediate feature map
        # VN: Lợi ích: Trích xuất đặc trưng cấp cao, đại diện cho nội dung ngữ nghĩa của ảnh
        # EN: Benefit: Extracts high-level features representing the semantic content of images

    # VN: Hàm content_loss, tính mất mát nội dung giữa ảnh nguồn và ảnh mục tiêu
    # EN: content_loss function, computes content loss between source and target images
    def content_loss(self, src, tgt):
        with torch.no_grad():  # VN: Tắt gradient để tối ưu hóa hiệu suất trong suy luận
            src_features = self.extract_features(src)  # VN: Trích xuất đặc trưng từ ảnh nguồn
                                                      # EN: Extract features from source image
            tgt_features = self.extract_features(tgt)  # VN: Trích xuất đặc trưng từ ảnh mục tiêu
                                                      # EN: Extract features from target image
        return F.mse_loss(src_features, tgt_features)  # VN: Tính mất mát MSE giữa đặc trưng nguồn và mục tiêu
                                                      # EN: Compute MSE loss between source and target features
        # VN: Lợi ích: Đảm bảo ảnh sinh giữ được nội dung ngữ nghĩa của ảnh nguồn
        # EN: Benefit: Ensures generated images retain semantic content of source images

# VN: Hàm tạo lịch trình beta cho quá trình khu Laura
# EN: Function to create beta schedule for diffusion process
def get_beta_schedule(timesteps=opt.step):
    beta_start = 0.0001  # VN: Giá trị beta khởi đầu (nhiễu nhỏ)
                         # EN: Starting beta value (small noise)
    beta_end = 0.02  # VN: Giá trị beta cuối (nhiễu lớn hơn)
                     # EN: Ending beta value (larger noise)
    return torch.linspace(beta_start, beta_end, timesteps)  # VN: Tạo lịch trình tuyến tính cho beta
                                                            # EN: Create linear schedule for beta
    # VN: Lợi ích: Lịch trình đơn giản, hiệu quả cho mô hình khuếch tán
    # EN: Benefit: Simple, effective schedule for diffusion models

# VN: Hàm thực hiện quá trình khuếch tán thuận (thêm nhiễu)
# EN: Function to perform forward diffusion (add noise)
def forward_diffusion(x_0, t, betas, device):
    sqrt_alphas_cumprod = torch.cumprod(1.0 - betas, dim=0).sqrt()  # VN: Tính hệ số tỷ lệ cho ảnh gốc
                                                                   # EN: Compute scaling factor for original image
    sqrt_one_minus_alphas_cumprod = (1.0 - torch.cumprod(1.0 - betas, dim=0)).sqrt()  # VN: Tính hệ số cho nhiễu
                                                                                     # EN: Compute scaling factor for noise
    
    noise = torch.randn_like(x_0).to(device)  # VN: Tạo nhiễu Gaussian
                                             # EN: Generate Gaussian noise
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t][:, None, None, None].to(device)  # VN: Chọn hệ số cho bước thời gian t
                                                                           # EN: Select scaling factor for timestep t
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t][:, None, None, None].to(device)
    
    x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise  # VN: Tính ảnh nhiễu tại bước t
                                                                                # EN: Compute noisy image at timestep t
    return x_t, noise  # VN: Trả về ảnh nhiễu và nhiễu
                       # EN: Return noisy image and noise
    # VN: Lợi ích: Thêm nhiễu hiệu quả trong một bước
    # EN: Benefit: Efficiently adds noise in one step

# VN: Hàm lấy mẫu từ mô hình khuếch tán có điều kiện
# EN: Function to sample from conditional diffusion model
def sample_conditional_diffusion(model, imgs_A, channels, img_size, betas, device):
    batch_size = imgs_A.size(0)  # VN: Lấy kích thước lô từ ảnh điều kiện
                                 # EN: Get batch size from condition images
    x_t = torch.randn(batch_size, channels, img_size, img_size).to(device)  # VN: Khởi tạo với nhiễu ngẫu nhiên
                                                                   # EN: Initialize with random noise
    for t in reversed(range(opt.step)):  # VN: Lặp ngược qua các bước thời gian
                                        # EN: Iterate backward through timesteps
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)  # VN: Tạo tensor bước thời gian
                                                                                # EN: Create timestep tensor
        with torch.no_grad():  # VN: Tắt gradient để suy luận
                              # EN: Disable gradients for inference
            pred_noise = model(x_t, t_batch, imgs_A)  # VN: Dự đoán nhiễu
                                                     # EN: Predict noise
        alpha = 1.0 - betas[t]  # VN: Tính alpha cho bước hiện tại
                                # EN: Compute alpha for current timestep
        x_t = (x_t - (1.0 - alpha) / torch.sqrt(1.0 - torch.cumprod(1.0 - betas, 0)[t]) * pred_noise) / torch.sqrt(alpha)  # VN: Cập nhật ảnh
                                                                                                                          # EN: Update image
        if t > 0:
            x_t += torch.sqrt(betas[t]) * torch.randn_like(x_t)  # VN: Thêm nhiễu trừ bước cuối
                                                                # EN: Add noise except at final step
    return x_t.clamp(-1, 1)  # VN: Chuẩn hóa đầu ra về [-1, 1]
                             # EN: Normalize output to [-1, 1]
    # VN: Lợi ích: Tạo ảnh chất lượng cao có điều kiện từ nhiễu
    # EN: Benefit: Generates high-quality conditioned images from noise