import torchvision.transforms as transforms
import torch
import random

# VN: Định nghĩa lớp ADAugment, dùng để tăng cường dữ liệu (data augmentation) với xác suất áp dụng
# EN: Define ADAugment class, used for data augmentation with a probability of application
class ADAugment(torch.nn.Module):
    # VN: Hàm khởi tạo với tham số xác suất áp dụng augmentation
    # EN: Constructor with probability parameter for applying augmentation
    def __init__(self, prob=0.0):
        super().__init__()  # VN: Gọi hàm khởi tạo của lớp cha nn.Module để đảm bảo tích hợp với PyTorch
                            # EN: Call the parent class nn.Module constructor to ensure PyTorch integration
        self.prob = prob  # VN: Lưu xác suất áp dụng augmentation (mặc định 0.0, tức là không áp dụng)
                          # EN: Store the probability of applying augmentation (default 0.0, i.e., no augmentation)
        # VN: Lợi ích: Cho phép kiểm soát mức độ augmentation thông qua tham số prob
        # EN: Benefit: Allows control over the degree of augmentation via the prob parameter
        
        # VN: Tạo pipeline tăng cường dữ liệu với các biến đổi nối tiếp
        # EN: Create a data augmentation pipeline with sequential transforms
        self.augment_pipeline = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),  # VN: Lật ngang ảnh với xác suất 100% khi được áp dụng
                                                    # EN: Horizontally flip the image with 100% probability when applied
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  # VN: Áp dụng biến đổi affine ngẫu nhiên (xoay tối đa 10 độ, dịch chuyển 10% kích thước ảnh)
                                                               # EN: Apply random affine transform (up to 10-degree rotation, 10% translation)
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)  # VN: Điều chỉnh ngẫu nhiên độ sáng, độ tương phản, độ bão hòa và sắc độ
                                                                                          # EN: Randomly adjust brightness, contrast, saturation, and hue
        ])  # VN: Lợi ích: Kết hợp nhiều biến đổi để tạo sự đa dạng dữ liệu, tăng khả năng tổng quát hóa của mô hình
            # EN: Benefit: Combines multiple transforms to create data diversity, improving model generalization

    # VN: Hàm forward, áp dụng pipeline tăng cường dữ liệu dựa trên xác suất
    # EN: Forward function, applies augmentation pipeline based on probability
    def forward(self, x):
        if random.random() < self.prob:  # VN: Kiểm tra ngẫu nhiên xem có áp dụng augmentation hay không dựa trên prob
                                         # EN: Randomly check whether to apply augmentation based on prob
            return self.augment_pipeline(x)  # VN: Áp dụng pipeline augmentation lên đầu vào x nếu điều kiện đúng
                                            # EN: Apply augmentation pipeline to input x if condition is met
        return x  # VN: Trả về đầu vào gốc nếu không áp dụng augmentation
                  # EN: Return original input if no augmentation is applied
        # VN: Lợi ích: Cho phép áp dụng augmentation có điều kiện, linh hoạt trong huấn luyện
        # EN: Benefit: Allows conditional application of augmentation, flexible for training