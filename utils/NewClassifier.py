import torch.nn as nn
import torch
from config import opt

# VN: Định nghĩa lớp Classifier, một mạng nơ-ron phân loại ảnh đơn giản
# EN: Define Classifier class, a simple neural network for image classification
class Classifier(nn.Module):
    # VN: Hàm khởi tạo cho mô hình phân loại
    # EN: Constructor for the classification model
    def __init__(self):
        super().__init__()  # VN: Gọi hàm khởi tạo của lớp cha nn.Module để tích hợp với PyTorch
                            # EN: Call the parent class nn.Module constructor for PyTorch integration
        
        # VN: Lớp chập đầu tiên: từ 3 kênh (RGB) sang 16 kênh
        # EN: First convolutional layer: from 3 channels (RGB) to 16 channels
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # VN: Chập 3x3, giữ nguyên kích thước không gian nhờ padding=1
                                             # EN: 3x3 conv, preserves spatial size with padding=1
        self.relu1 = nn.ReLU()  # VN: Hàm kích hoạt ReLU để thêm phi tuyến tính
                                # EN: ReLU activation to add non-linearity
        self.pool1 = nn.MaxPool2d(2)  # VN: Max pooling 2x2 để giảm kích thước không gian xuống một nửa
                                      # EN: 2x2 max pooling to reduce spatial size by half
        # VN: Lợi ích: Khối conv1-relu1-pool1 trích xuất đặc trưng cơ bản từ ảnh
        # EN: Benefit: conv1-relu1-pool1 block extracts basic features from the image

        # VN: Lớp chập thứ hai: từ 16 kênh sang 32 kênh
        # EN: Second convolutional layer: from 16 channels to 32 channels
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # VN: Chập 3x3, tiếp tục giữ kích thước không gian
                                              # EN: 3x3 conv, continues to preserve spatial size
        self.relu2 = nn.ReLU()  # VN: ReLU thứ hai để thêm phi tuyến tính
                                # EN: Second ReLU for non-linearity
        self.pool2 = nn.MaxPool2d(2)  # VN: Max pooling thứ hai, giảm kích thước không gian lần nữa
                                      # EN: Second max pooling, further reduces spatial size
        # VN: Lợi ích: Khối conv2-relu2-pool2 trích xuất đặc trưng cấp cao hơn
        # EN: Benefit: conv2-relu2-pool2 block extracts higher-level features

        self.flatten = nn.Flatten()  # VN: Lớp làm phẳng để chuyển bản đồ đặc trưng thành vector
                                     # EN: Flatten layer to convert feature map to a vector
        # VN: Lợi ích: Chuẩn bị dữ liệu cho các tầng tuyến tính
        # EN: Benefit: Prepares data for fully connected layers

        # VN: Tính kích thước đầu vào cho tầng tuyến tính dựa trên kích thước ảnh sau hai lần pooling
        # EN: Calculate input size for linear layer based on image size after two poolings
        self.fc1 = nn.Linear(32 * (opt.img_size // 4) * (opt.img_size // 4), 128)  # VN: Tầng tuyến tính từ số đặc trưng phẳng sang 128
                                                                           # EN: Linear layer from flattened features to 128 units
        self.relu3 = nn.ReLU()  # VN: ReLU thứ ba để thêm phi tuyến tính
                                # EN: Third ReLU for non-linearity
        # VN: Lợi ích: Tầng fc1-relu3 giảm chiều dữ liệu và thêm phi tuyến tính
        # EN: Benefit: fc1-relu3 reduces dimensionality and adds non-linearity

        self.fc2 = nn.Linear(128, 10)  # VN: Tầng tuyến tính cuối, ánh xạ sang 10 lớp (giả định 10 lớp phân loại)
                                       # EN: Final linear layer, maps to 10 classes (assuming 10-class classification)
        # VN: Lợi ích: Tạo đầu ra phân loại phù hợp cho bài toán phân loại đa lớp
        # EN: Benefit: Produces classification output suitable for multi-class classification

    # VN: Hàm forward, xử lý đầu vào và trả về đầu ra phân loại
    # EN: Forward function, processes input and returns classification output
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))  # VN: Xử lý qua khối conv1-relu1-pool1
                                                  # EN: Process through conv1-relu1-pool1 block
        x = self.pool2(self.relu2(self.conv2(x)))  # VN: Xử lý qua khối conv2-relu2-pool2
                                                   # EN: Process through conv2-relu2-pool2 block
        x = self.flatten(x)  # VN: Làm phẳng bản đồ đặc trưng thành vector
                             # EN: Flatten feature map to a vector
        x = self.relu3(self.fc1(x))  # VN: Xử lý qua tầng tuyến tính fc1 và ReLU
                                     # EN: Process through fc1 linear layer and ReLU
        return self.fc2(x)  # VN: Tạo đầu ra cuối cùng qua tầng fc2
                            # EN: Produce final output through fc2 layer
        # VN: Lợi ích: Quy trình đơn giản, hiệu quả để phân loại ảnh với đầu ra là điểm số lớp
        # EN: Benefit: Simple, efficient pipeline for image classification with class scores as output

    # VN: Hàm feature_vector, trích xuất vector đặc trưng cho SemanticConsistencyModule
    # EN: feature_vector function, extracts feature vector for SemanticConsistencyModule
    def feature_vector(self, x):
        # VN: Hỗ trợ SemanticConsistencyModule bằng cách trích xuất đặc trưng trung gian
        # EN: Support SemanticConsistencyModule by extracting intermediate features
        with torch.no_grad():  # VN: Tắt tính toán gradient để tiết kiệm tài nguyên trong suy luận
                              # EN: Disable gradient computation to save resources during inference
            x = self.pool1(self.relu1(self.conv1(x)))  # VN: Xử lý qua khối conv1-relu1-pool1
                                                       # EN: Process through conv1-relu1-pool1 block
            x = self.pool2(self.relu2(self.conv2(x)))  # VN: Xử lý qua khối conv2-relu2-pool2
                                                       # EN: Process through conv2-relu2-pool2 block
            return self.flatten(x)  # VN: Trả về vector đặc trưng phẳng
                                    # EN: Return flattened feature vector
        # VN: Lợi ích: Cung cấp đặc trưng trung gian để tính loss ngữ nghĩa, hỗ trợ các tác vụ như chuyển đổi phong cách
        # EN: Benefit: Provides intermediate features for semantic loss computation, supporting tasks like style transfer