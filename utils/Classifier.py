import torch.nn as nn
from config import opt

# VN: Định nghĩa lớp Classifier, một mạng nơ-ron phân loại ảnh
# EN: Define Classifier class, a neural network for image classification
class Classifier(nn.Module):
    # VN: Hàm khởi tạo cho mô hình phân loại
    # EN: Constructor for the classification model
    def __init__(self):
        super(Classifier, self).__init__()  # VN: Gọi hàm khởi tạo của lớp cha nn.Module để tích hợp với PyTorch
                                            # EN: Call the parent class nn.Module constructor for PyTorch integration
        
        # VN: Hàm block tạo một khối mạng với Conv2d, LeakyReLU và tùy chọn chuẩn hóa
        # EN: block function creates a network block with Conv2d, LeakyReLU, and optional normalization
        def block(in_features, out_features, normalization=True):
            """Classifier block"""
            # VN: Tạo danh sách các tầng: chập 3x3, LeakyReLU và chuẩn hóa (nếu bật)
            # EN: Create list of layers: 3x3 convolution, LeakyReLU, and normalization (if enabled)
            layers = [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),  # VN: Lớp chập giảm kích thước không gian (stride=2), giữ số kênh
                                                                       # EN: Conv layer reduces spatial size (stride=2), maintains channels
                nn.LeakyReLU(0.2, inplace=True)  # VN: LeakyReLU với độ dốc 0.2 để tránh chết nơ-ron
                                                 # EN: LeakyReLU with 0.2 slope to avoid dead neurons
            ]
            if normalization:  # VN: Nếu bật chuẩn hóa, thêm InstanceNorm2d
                layers.append(nn.InstanceNorm2d(out_features))  # VN: Chuẩn hóa instance để ổn định huấn luyện
                                                              # EN: Instance normalization to stabilize training
            # VN: Lợi ích: Khối này cung cấp mô-đun tái sử dụng để xây dựng mạng sâu
            # EN: Benefit: This block provides a reusable module for building deep networks
            return layers

        # VN: Xây dựng mô hình chính với các khối liên tiếp
        # EN: Build the main model with sequential blocks
        self.model = nn.Sequential(
            *block(opt.channels, 64, normalization=False),  # VN: Khối đầu tiên, không chuẩn hóa, chuyển từ số kênh đầu vào sang 64
                                                          # EN: First block, no normalization, maps from input channels to 64
            *block(64, 128),  # VN: Khối thứ hai, tăng kênh lên 128
                              # EN: Second block, increases channels to 128
            *block(128, 256),  # VN: Khối thứ ba, tăng kênh lên 256
                               # EN: Third block, increases channels to 256
            *block(256, 512)  # VN: Khối thứ tư, tăng kênh lên 512
                              # EN: Fourth block, increases channels to 512
        )  # VN: Lợi ích: Kiến trúc phân cấp giúp trích xuất đặc trưng từ thấp đến cao
           # EN: Benefit: Hierarchical architecture extracts features from low to high level

        # VN: Tính kích thước đầu vào cho tầng tuyến tính dựa trên kích thước ảnh
        # EN: Calculate input size for linear layer based on image size
        input_size = opt.img_size // 2 ** 4  # VN: Kích thước không gian sau 4 lần giảm (stride=2)
                                             # EN: Spatial size after 4 downsamplings (stride=2)
        # VN: Tầng đầu ra với Linear và Softmax
        # EN: Output layer with Linear and Softmax
        self.output_layer = nn.Sequential(
            nn.Linear(512 * input_size ** 2, opt.n_classes),  # VN: Lớp tuyến tính ánh xạ đặc trưng sang số lớp
                                                             # EN: Linear layer maps features to number of classes
            nn.Softmax(dim=1)  # VN: Softmax để tạo phân phối xác suất trên các lớp
                               # EN: Softmax to produce probability distribution over classes
        )  # VN: Lợi ích: Chuyển đổi đặc trưng thành xác suất phân loại, dễ sử dụng cho bài toán phân loại
           # EN: Benefit: Converts features to classification probabilities, suitable for classification tasks

    # VN: Hàm forward, xử lý đầu vào và trả về nhãn phân loại
    # EN: Forward function, processes input and returns classification labels
    def forward(self, img):
        feature_repr = self.model(img)  # VN: Xử lý ảnh qua các khối chập để tạo biểu diễn đặc trưng
                                        # EN: Process image through conv blocks to generate feature representation
        feature_repr = feature_repr.view(feature_repr.size(0), -1)  # VN: Chuyển bản đồ đặc trưng thành vector phẳng
                                                                   # EN: Flatten feature map to a vector
        # VN: Lợi ích: Chuẩn bị đặc trưng cho tầng tuyến tính
        # EN: Benefit: Prepares features for the linear layer
        label = self.output_layer(feature_repr)  # VN: Áp dụng tầng đầu ra để tạo nhãn
                                                # EN: Apply output layer to produce labels
        return label  # VN: Trả về phân phối xác suất trên các lớp
                      # EN: Return probability distribution over classes
        # VN: Lợi ích: Cung cấp đầu ra phân loại trực tiếp, dễ tích hợp vào pipeline huấn luyện
        # EN: Benefit: Provides direct classification output, easy to integrate into training pipeline