import os
import numpy as np
import json
import itertools
import shutil
import lpips
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR

from utils.mnistm import MNISTM
from utils.NewClassifier import Classifier  
from utils.NewGenerator import Generator
from utils.NewDiscriminator import DomainConfusionDiscriminator
from utils.ConditionalDiffusionModel import ConditionalDiffusionModel, get_beta_schedule, forward_diffusion, sample_conditional_diffusion, DiffusionStyleAnchor, SemanticConsistencyModule
from utils.EMA import EMA
from config import opt

# VN: Xóa các file và thư mục cũ nếu tồn tại, tạo thư mục mới để lưu kết quả
# EN: Remove old files and directories if they exist, create new directories for results
if os.path.exists("training_log.json"):
    os.remove("training_log.json")  # VN: Xóa file log cũ để bắt đầu mới
                                    # EN: Remove old log file to start fresh
if os.path.exists("imagesss"):
    shutil.rmtree("imagesss")  # VN: Xóa thư mục ảnh cũ
                               # EN: Remove old images directory
os.makedirs("imagesss", exist_ok=True)  # VN: Tạo thư mục mới để lưu ảnh mẫu
                                        # EN: Create new directory for sample images
if os.path.exists("diffusion_features"):
    shutil.rmtree("diffusion_features")  # VN: Xóa thư mục đặc trưng khuếch tán cũ
                                        # EN: Remove old diffusion features directory
os.makedirs("diffusion_features", exist_ok=True)  # VN: Tạo thư mục mới để lưu đặc trưng khuếch tán
                                                 # EN: Create new directory for diffusion features
if os.path.exists("checkpoint"):
    shutil.rmtree("checkpoint")  # VN: Xóa thư mục checkpoint cũ
                                 # EN: Remove old checkpoint directory
os.makedirs("checkpoint", exist_ok=True)  # VN: Tạo thư mục mới để lưu checkpoint
                                         # EN: Create new directory for checkpoints
# VN: Lợi ích: Đảm bảo môi trường sạch trước khi huấn luyện, tránh xung đột với kết quả cũ
# EN: Benefit: Ensures a clean environment before training, avoiding conflicts with old results

# VN: Kiểm tra và thiết lập thiết bị tính toán (GPU hoặc CPU)
# EN: Check and set computing device (GPU or CPU)
cuda = True if torch.cuda.is_available() else False  # VN: Kiểm tra xem có GPU không
                                                     # EN: Check if GPU is available
device = torch.device(f"cuda:{opt.n_gpu}" if torch.cuda.is_available() else "cpu")  # VN: Chọn thiết bị dựa trên cấu hình GPU
                                                                            # EN: Select device based on GPU configuration
if cuda:
    torch.cuda.set_device(opt.n_gpu)  # VN: Đặt GPU cụ thể nếu có nhiều GPU
                                      # EN: Set specific GPU if multiple GPUs are available
# VN: Lợi ích: Tận dụng GPU để tăng tốc huấn luyện nếu khả dụng, đảm bảo tính linh hoạt
# EN: Benefit: Leverages GPU for faster training if available, ensuring flexibility

# VN: Hàm khởi tạo trọng số cho các tầng mạng
# EN: Function to initialize weights for network layers
def weights_init_normal(m):
    classname = m.__class__.__name__  # VN: Lấy tên lớp của mô-đun
                                      # EN: Get the class name of the module
    if classname.find("Conv") != -1:  # VN: Nếu là lớp chập
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)  # VN: Khởi tạo trọng số chập theo phân phối chuẩn (mean=0, std=0.02)
                                                 # EN: Initialize conv weights with normal distribution (mean=0, std=0.02)
    elif classname.find("BatchNorm") != -1:  # VN: Nếu là lớp chuẩn hóa lô
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)  # VN: Khởi tạo trọng số chuẩn hóa lô (mean=1, std=0.02)
                                                 # EN: Initialize batch norm weights (mean=1, std=0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)  # VN: Đặt bias của chuẩn hóa lô bằng 0
                                                  # EN: Set batch norm bias to 0
# VN: Lợi ích: Khởi tạo trọng số đồng nhất giúp ổn định huấn luyện ban đầu
# EN: Benefit: Uniform weight initialization stabilizes initial training

# VN: Hàm tính loss nhất quán ngữ nghĩa giữa source và target
# EN: Function to compute semantic consistency loss between source and target
def semantic_consistency_loss(logits_src, logits_tgt):
    p_src = torch.softmax(logits_src, dim=1)  # VN: Chuyển logits source thành xác suất
                                              # EN: Convert source logits to probabilities
    log_p_tgt = torch.log_softmax(logits_tgt, dim=1)  # VN: Tính log softmax của logits target
                                                      # EN: Compute log softmax of target logits
    return F.kl_div(log_p_tgt, p_src, reduction='batchmean')  # VN: Tính KL divergence giữa target và source
                                                             # EN: Compute KL divergence between target and source
# VN: Lợi ích: Đảm bảo tính nhất quán ngữ nghĩa giữa ảnh gốc và ảnh sinh ra
# EN: Benefit: Ensures semantic consistency between original and generated images

# VN: Hàm tiền xử lý ảnh để chuẩn hóa giá trị pixel về [-1, 1]
# EN: Function to preprocess images, normalizes pixel values to [-1, 1]
def preprocess(img):
    return img * 2 - 1  # VN: Chuyển giá trị pixel từ [0, 1] sang [-1, 1]
                        # EN: Convert pixel values from [0, 1] to [-1, 1]
# VN: Lợi ích: Chuẩn hóa đầu vào cho các hàm mất mát như LPIPS
# EN: Benefit: Normalizes input for loss functions like LPIPS

# VN: Hàm tạo ảnh màu từ MNIST grayscale
# EN: Function to colorize MNIST grayscale images
def colorize_mnist(img):
    img = transforms.Resize(opt.img_size)(img)  # VN: Thay đổi kích thước ảnh thành opt.img_size
                                               # EN: Resize image to opt.img_size
    img = transforms.ToTensor()(img)  # VN: Chuyển ảnh thành tensor
                                      # EN: Convert image to tensor
    r, g, b = torch.rand(3)  # VN: Tạo ngẫu nhiên hệ số màu cho RGB
                             # EN: Generate random color coefficients for RGB
    img_rgb = torch.cat([
        img * r,
        img * g,
        img * b
    ], dim=0)  # VN: Tạo ảnh RGB bằng cách nhân kênh grayscale với hệ số màu
               # EN: Create RGB image by multiplying grayscale channel with color coefficients
    return transforms.Normalize([0.5]*3, [0.5]*3)(img_rgb)  # VN: Chuẩn hóa ảnh RGB về [-1, 1]
                                                    # EN: Normalize RGB image to [-1, 1]
# VN: Lợi ích: Tạo ảnh màu từ MNIST grayscale để phù hợp với miền mục tiêu
# EN: Benefit: Creates colored images from MNIST grayscale to match target domain

# VN: Khởi tạo các hàm mất mát
# EN: Initialize loss functions
adversarial_loss = torch.nn.MSELoss()  # VN: Hàm mất mát MSE cho huấn luyện đối kháng
                                       # EN: MSE loss function for adversarial training
task_loss = torch.nn.CrossEntropyLoss()  # VN: Hàm mất mát CrossEntropy cho phân loại
                                         # EN: CrossEntropy loss for classification
loss_fn = lpips.LPIPS(net='vgg').eval().to(device)  # VN: Hàm mất mát LPIPS (VGG) để đo độ giống về nhận thức, đặt ở chế độ eval
                                                    # EN: LPIPS (VGG) loss for perceptual similarity, set to eval mode
# VN: Lợi ích: Kết hợp nhiều loại mất mát để tối ưu hóa cả chất lượng hình ảnh và hiệu suất phân loại
# EN: Benefit: Combines multiple loss types to optimize both image quality and classification performance

# VN: Khởi tạo trọng số mất mát với chiến lược học theo chương trình
# EN: Initialize loss weights with curriculum learning strategy
lambda_adv = 1  # VN: Trọng số cho mất mát đối kháng
                # EN: Weight for adversarial loss
lambda_style = 0.5  # VN: Trọng số cho mất mát phong cách
                    # EN: Weight for style loss
lambda_task = 1  # VN: Trọng số cho mất mát phân loại
                 # EN: Weight for classification loss
lambda_content = 0.5  # VN: Trọng số cho mất mát nội dung
                      # EN: Weight for content loss
lambda_semantic = 1  # VN: Trọng số cho mất mát nhất quán ngữ nghĩa
                     # EN: Weight for semantic consistency loss
lambda_domain = 0.5  # VN: Trọng số cho mất mát phân biệt miền
                     # EN: Weight for domain confusion loss
# VN: Lợi ích: Điều chỉnh trọng số giúp cân bằng các mục tiêu huấn luyện
# EN: Benefit: Adjustable weights balance different training objectives

# VN: Khởi tạo các mô hình
# EN: Initialize models
generator = Generator().to(device)  # VN: Khởi tạo Generator và chuyển sang thiết bị
                                    # EN: Initialize Generator and move to device
discriminator = DomainConfusionDiscriminator().to(device)  # VN: Khởi tạo Discriminator và chuyển sang thiết bị
                                                  # EN: Initialize Discriminator and move to device
classifier = Classifier().to(device)  # VN: Khởi tạo Classifier và chuyển sang thiết bị
                                      # EN: Initialize Classifier and move to device
betas = get_beta_schedule().to(device)  # VN: Tạo lịch trình beta cho mô hình khuếch tán
                                        # EN: Create beta schedule for diffusion model
diffusion_model = ConditionalDiffusionModel(channels=opt.channels, img_size=opt.img_size).to(device)  # VN: Khởi tạo mô hình khuếch tán và chuyển sang thiết bị
                                                                              # EN: Initialize diffusion model and move to device
# VN: Lợi ích: Tích hợp nhiều mô hình để hỗ trợ huấn luyện đối kháng và khuếch tán
# EN: Benefit: Integrates multiple models for adversarial and diffusion training

# VN: Áp dụng khởi tạo trọng số
# EN: Apply weight initialization
generator.apply(weights_init_normal)  # VN: Khởi tạo trọng số cho Generator
                                     # EN: Initialize weights for Generator
discriminator.apply(weights_init_normal)  # VN: Khởi tạo trọng số cho Discriminator
                                          # EN: Initialize weights for Discriminator
classifier.apply(weights_init_normal)  # VN: Khởi tạo trọng số cho Classifier
                                       # EN: Initialize weights for Classifier
diffusion_model.apply(weights_init_normal)  # VN: Khởi tạo trọng số cho mô hình khuếch tán
                                           # EN: Initialize weights for diffusion model
# VN: Lợi ích: Đảm bảo các mô hình bắt đầu với trọng số ổn định
# EN: Benefit: Ensures models start with stable weights

# VN: Khởi tạo các mô-đun DGAA
# EN: Initialize DGAA modules
style_anchor = DiffusionStyleAnchor(diffusion_model).eval().to(device)  # VN: Khởi tạo StyleAnchor, đặt ở chế độ eval
                                                               # EN: Initialize StyleAnchor, set to eval mode
semantic_module = SemanticConsistencyModule(classifier).to(device)  # VN: Khởi tạo SemanticConsistencyModule
                                                                   # EN: Initialize SemanticConsistencyModule
# VN: Lợi ích: Tăng cường phong cách và nhất quán ngữ nghĩa cho ảnh sinh ra
# EN: Benefit: Enhances style and semantic consistency of generated images

# VN: Khởi tạo EMA cho Generator
# EN: Initialize EMA for Generator
ema_g = EMA(generator, decay=0.999)  # VN: Khởi tạo EMA với decay 0.999 để làm mượt trọng số Generator
                                     # EN: Initialize EMA with decay 0.999 to smooth Generator weights
ema_g.register()  # VN: Đăng ký tham số shadow cho EMA
                  # EN: Register shadow parameters for EMA
# VN: Lợi ích: EMA giúp cải thiện độ ổn định và chất lượng của Generator trong suy luận
# EN: Benefit: EMA improves Generator stability and quality during inference

# VN: Khởi tạo các bộ tối ưu hóa
# EN: Initialize optimizers
optimizer_G = torch.optim.AdamW(
    itertools.chain(generator.parameters(), classifier.parameters()), 
    lr=opt.lr, 
    betas=(opt.b1, opt.b2),
    weight_decay=1e-5
)  # VN: Tối ưu hóa AdamW cho Generator và Classifier, kết hợp tham số
   # EN: AdamW optimizer for Generator and Classifier, combining parameters
optimizer_D = torch.optim.AdamW(
    discriminator.parameters(), 
    lr=opt.lr, 
    betas=(opt.b1, opt.b2)
)  # VN: Tối ưu hóa AdamW cho Discriminator
   # EN: AdamW optimizer for Discriminator
optimizer_diff = torch.optim.AdamW(
    diffusion_model.parameters(), 
    lr=opt.lr
)  # VN: Tối ưu hóa AdamW cho mô hình khuếch tán
   # EN: AdamW optimizer for diffusion model
# VN: Lợi ích: Sử dụng AdamW với weight decay để cải thiện tổng quát hóa
# EN: Benefit: Uses AdamW with weight decay to improve generalization

# VN: Khởi tạo các bộ điều chỉnh tốc độ học
# EN: Initialize learning rate schedulers
scheduler_G = ExponentialLR(optimizer_G, gamma=0.98)  # VN: Giảm tốc độ học theo cấp số nhân cho Generator
                                                     # EN: Exponential decay for Generator learning rate
scheduler_D = ExponentialLR(optimizer_D, gamma=0.98)  # VN: Giảm tốc độ học theo cấp số nhân cho Discriminator
                                                     # EN: Exponential decay for Discriminator learning rate
scheduler_diff = ExponentialLR(optimizer_diff, gamma=0.98)  # VN: Giảm tốc độ học theo cấp số nhân cho mô hình khuếch tán
                                                           # EN: Exponential decay for diffusion model learning rate
# VN: Lợi ích: Giảm dần tốc độ học để ổn định huấn luyện về cuối
# EN: Benefit: Gradually reduces learning rate to stabilize training towards the end

# VN: Cấu hình DataLoader cho miền A (MNIST)
# EN: Configure DataLoader for domain A (MNIST)
dataloader_A = torch.utils.data.DataLoader(
    datasets.MNIST(
        "/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f41/Dat/Teo/SIC/PyTorch-GAN/data/mnist",
        train=True,
        download=False,
        transform=colorize_mnist,  # VN: Áp dụng hàm colorize_mnist để tạo ảnh màu
                                   # EN: Apply colorize_mnist to create colored images
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)  # VN: Lợi ích: Tải dữ liệu MNIST hiệu quả với batch size và shuffling
   # EN: Benefit: Efficiently loads MNIST data with batch size and shuffling

# VN: Cấu hình DataLoader cho miền B (MNIST-M)
# EN: Configure DataLoader for domain B (MNIST-M)
dataloader_B = torch.utils.data.DataLoader(
    MNISTM(
        "/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f41/Dat/Teo/SIC/PyTorch-GAN/data/mnistm",
        train=True,
        download=False,
        transform=transforms.Compose(
            [
                transforms.Resize(opt.img_size),  # VN: Thay đổi kích thước ảnh
                                                  # EN: Resize images
                transforms.ToTensor(),  # VN: Chuyển thành tensor
                                        # EN: Convert to tensor
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # VN: Chuẩn hóa về [-1, 1]
                                                                  # EN: Normalize to [-1, 1]
            ]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)  # VN: Lợi ích: Tải dữ liệu MNIST-M với tiền xử lý phù hợp
   # EN: Benefit: Loads MNIST-M data with appropriate preprocessing

# VN: Định nghĩa kiểu tensor dựa trên thiết bị
# EN: Define tensor types based on device
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor  # VN: Tensor float cho thiết bị
                                                                    # EN: Float tensor for device
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor  # VN: Tensor long cho thiết bị
                                                                  # EN: Long tensor for device
# VN: Lợi ích: Đảm bảo tensor tương thích với thiết bị tính toán
# EN: Benefit: Ensures tensors are compatible with computing device

# VN: Khởi tạo các biến theo dõi huấn luyện
# EN: Initialize training tracking variables
task_performance = []  # VN: Danh sách lưu hiệu suất phân loại trên ảnh sinh
                       # EN: List to store classification performance on generated images
target_performance = []  # VN: Danh sách lưu hiệu suất phân loại trên ảnh mục tiêu
                         # EN: List to store classification performance on target images
log_data = []  # VN: Danh sách lưu log huấn luyện
               # EN: List to store training logs
best = float('inf')  # VN: Theo dõi giá trị tốt nhất (sẽ được cập nhật thành độ chính xác tốt nhất)
                     # EN: Track best value (will be updated to best accuracy)
step = 0  # VN: Biến đếm bước huấn luyện
          # EN: Training step counter
# VN: Lợi ích: Theo dõi và lưu trữ thông tin huấn luyện để đánh giá và lưu checkpoint
# EN: Benefit: Tracks and stores training information for evaluation and checkpointing

# =============================================
# CLASSIFIER 
# =============================================
# VN: Khởi tạo tối ưu hóa và scheduler cho Classifier
# EN: Initialize optimizer and scheduler for Classifier
optimizer_clf = torch.optim.Adam(classifier.parameters(), lr=opt.lr)  # VN: Tối ưu hóa Adam cho Classifier
                                                             # EN: Adam optimizer for Classifier
scheduler_clf = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_clf, 
    mode='max', 
    factor=0.5, 
    patience=5
)  # VN: Scheduler giảm tốc độ học nếu độ chính xác không cải thiện sau 5 epoch
   # EN: Scheduler reduces learning rate if accuracy does not improve after 5 epochs
# VN: Lợi ích: Tối ưu hóa và điều chỉnh tốc độ học linh hoạt để cải thiện hiệu suất Classifier
# EN: Benefit: Flexible optimization and learning rate adjustment to improve Classifier performance

best_acc = 0.0  # VN: Theo dõi độ chính xác tốt nhất của Classifier
                # EN: Track best accuracy of Classifier
classifier.train()  # VN: Đặt Classifier ở chế độ huấn luyện
                    # EN: Set Classifier to training mode

# VN: Vòng lặp huấn luyện trước Classifier trên dữ liệu MNIST
# EN: Loop for pre-training Classifier on MNIST data
for epoch in tqdm(range(30), desc="Classifier Pre-training"):
    epoch_loss = 0.0  # VN: Theo dõi mất mát trung bình của epoch
                      # EN: Track average loss of epoch
    epoch_correct = 0  # VN: Theo dõi số dự đoán đúng
                       # EN: Track number of correct predictions
    total_samples = 0  # VN: Theo dõi tổng số mẫu
                       # EN: Track total number of samples
    
    for i, (imgs, labels) in enumerate(dataloader_A):  # VN: Duyệt qua các batch của miền A
        imgs, labels = imgs.to(device), labels.to(device)  # VN: Chuyển dữ liệu sang thiết bị
                                                           # EN: Move data to device
        
        if imgs.dtype == torch.uint8:  # VN: Kiểm tra và chuyển đổi kiểu dữ liệu nếu cần
            imgs = imgs.float() / 255.0  # VN: Chuyển ảnh uint8 sang float và chuẩn hóa
                                         # EN: Convert uint8 images to float and normalize
        
        optimizer_clf.zero_grad()  # VN: Xóa gradient cũ
                                  # EN: Clear old gradients
        outputs = classifier(imgs)  # VN: Dự đoán nhãn từ Classifier
                                   # EN: Predict labels from Classifier
        loss = task_loss(outputs, labels)  # VN: Tính mất mát CrossEntropy
                                           # EN: Compute CrossEntropy loss
        loss.backward()  # VN: Tính gradient
                         # EN: Compute gradients
        
        # VN: Cắt gradient để ổn định huấn luyện
        # EN: Clip gradients for training stability
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)  # VN: Giới hạn norm gradient ở 1.0
                                                                      # EN: Limit gradient norm to 1.0
        
        optimizer_clf.step()  # VN: Cập nhật trọng số Classifier
                             # EN: Update Classifier weights
        
        preds = torch.argmax(outputs, dim=1)  # VN: Lấy nhãn dự đoán
                                              # EN: Get predicted labels
        correct = (preds == labels).sum().item()  # VN: Tính số dự đoán đúng
                                                  # EN: Calculate number of correct predictions
        total = labels.size(0)  # VN: Lấy kích thước batch
                                # EN: Get batch size
        
        epoch_loss += loss.item() * total  # VN: Cộng dồn mất mát
                                           # EN: Accumulate loss
        epoch_correct += correct  # VN: Cộng dồn số dự đoán đúng
                                 # EN: Accumulate correct predictions
        total_samples += total  # VN: Cộng dồn số mẫu
                               # EN: Accumulate total samples
        
        if i % 100 == 0:  # VN: In thông tin mỗi 100 batch
            batch_acc = correct / total  # VN: Tính độ chính xác batch
                                         # EN: Calculate batch accuracy
            print(f"Epoch {epoch}/{30} | Batch {i}/{len(dataloader_A)} | "
                  f"Loss: {loss.item():.4f} | Acc: {batch_acc:.4f}")  # VN: In mất mát và độ chính xác
                                                               # EN: Print loss and accuracy
    
    epoch_loss /= total_samples  # VN: Tính mất mát trung bình của epoch
                                 # EN: Compute average epoch loss
    epoch_acc = epoch_correct / total_samples  # VN: Tính độ chính xác trung bình của epoch
                                               # EN: Compute average epoch accuracy
    
    # VN: Cập nhật scheduler dựa trên độ chính xác
    # EN: Update scheduler based on accuracy
    old_lr = optimizer_clf.param_groups[0]['lr']  # VN: Lưu tốc độ học cũ
                                                 # EN: Store old learning rate
    scheduler_clf.step(epoch_acc)  # VN: Cập nhật tốc độ học
                                  # EN: Update learning rate
    new_lr = optimizer_clf.param_groups[0]['lr']  # VN: Lấy tốc độ học mới
                                                 # EN: Get new learning rate
    
    if new_lr < old_lr:  # VN: Thông báo nếu tốc độ học giảm
        print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")  # VN: In thông tin giảm tốc độ học
                                                                  # EN: Print learning rate reduction
    
    if epoch_acc > best_acc:  # VN: Lưu mô hình nếu độ chính xác cải thiện
        best_acc = epoch_acc  # VN: Cập nhật độ chính xác tốt nhất
                              # EN: Update best accuracy
        torch.save(classifier.state_dict(), 'best_classifier.pth')  # VN: Lưu trạng thái mô hình tốt nhất
                                                           # EN: Save best model state
        print(f"Saved best model with acc: {best_acc:.4f}")  # VN: In thông tin lưu mô hình
                                                             # EN: Print model save information
    
    print(f"Epoch {epoch} Summary | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Best Acc: {best_acc:.4f}")  # VN: In tóm tắt epoch
                                                                                             # EN: Print epoch summary
    print(f"Current LR: {optimizer_clf.param_groups[0]['lr']}")  # VN: In tốc độ học hiện tại
                                                                # EN: Print current learning rate

print(f"Final Best Accuracy: {best_acc:.4f}")  # VN: In độ chính xác tốt nhất cuối cùng
                                              # EN: Print final best accuracy

# VN: Huấn luyện trước mô hình khuếch tán trên miền mục tiêu (MNIST-M)
# EN: Pre-train diffusion model on target domain (MNIST-M)
if not os.path.exists('targetdiff.pth'):  # VN: Kiểm tra xem mô hình khuếch tán đã được lưu chưa
    # VN: Lấy nhãn từ dataloader_A để khớp lớp
    # EN: Get labels from dataloader_A for class matching
    targets_A = dataloader_A.dataset.targets.numpy()  # VN: Chuyển nhãn miền A thành mảng numpy
                                                     # EN: Convert domain A labels to numpy array
    
    for epoch in tqdm(range(1000), desc="Diffusion Pre-training"):  # VN: Huấn luyện 1000 epoch
        for (imgs_B, labels_B) in dataloader_B:  # VN: Duyệt qua dữ liệu miền B
            imgs_B = imgs_B.to(device)  # VN: Chuyển ảnh sang thiết bị
                                        # EN: Move images to device
            labels_B = labels_B.to(device)  # VN: Chuyển nhãn sang thiết bị
                                            # EN: Move labels to device
            t = torch.randint(0, 1000, (imgs_B.size(0),), device=device).long()  # VN: Tạo bước thời gian ngẫu nhiên
                                                                         # EN: Generate random timestep
            
            source_indices = []  # VN: Danh sách lưu chỉ số ảnh miền A
                                 # EN: List to store domain A image indices
            for label in labels_B:  # VN: Duyệt qua nhãn của miền B
                # VN: Tìm chỉ số ảnh có cùng nhãn trong miền A
                # EN: Find indices with same label in domain A
                same_class_indices = np.where(targets_A == label.item())[0]
                if len(same_class_indices) == 0:  # VN: Nếu không tìm thấy, chọn ngẫu nhiên
                    idx = random.randint(0, len(dataloader_A.dataset)-1)  # VN: Chọn chỉ số ngẫu nhiên
                                                                 # EN: Choose random index
                else:
                    idx = random.choice(same_class_indices)  # VN: Chọn ngẫu nhiên từ các chỉ số cùng lớp
                                                            # EN: Randomly choose from same-class indices
                source_indices.append(idx)
            
            # VN: Lấy ảnh từ miền A dựa trên chỉ số
            # EN: Get images from domain A based on indices
            imgs_A = torch.stack([dataloader_A.dataset[i][0] for i in source_indices]).to(device)
            
            # VN: Thực hiện khuếch tán thuận
            # EN: Perform forward diffusion
            x_t, noise = forward_diffusion(imgs_B, t, betas, device)  # VN: Tạo ảnh nhiễu và nhiễu
                                                              # EN: Generate noisy image and noise
            pred_noise = diffusion_model(x_t, t, imgs_A)  # VN: Dự đoán nhiễu từ mô hình khuếch tán
                                                  # EN: Predict noise from diffusion model
            
            loss = nn.MSELoss()(pred_noise, noise)  # VN: Tính mất mát MSE giữa nhiễu dự đoán và thực tế
                                                    # EN: Compute MSE loss between predicted and actual noise
            optimizer_diff.zero_grad()  # VN: Xóa gradient cũ
                                        # EN: Clear old gradients
            loss.backward()  # VN: Tính gradient
                             # EN: Compute gradients
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=1.0)  # VN: Cắt gradient
                                                                                # EN: Clip gradients
            optimizer_diff.step()  # VN: Cập nhật trọng số
                                   # EN: Update weights
        
        print(f"Diffusion Pre-train Epoch {epoch} Loss: {loss.item():.4f}")  # VN: In mất mát mỗi epoch
                                                                    # EN: Print loss per epoch
        scheduler_diff.step()  # VN: Cập nhật tốc độ học
                              # EN: Update learning rate
    
    torch.save(diffusion_model.state_dict(), 'targetdiff.pth')  # VN: Lưu trạng thái mô hình khuếch tán
                                                                # EN: Save diffusion model state
# VN: Lợi ích: Huấn luyện trước mô hình khuếch tán giúp cải thiện chất lượng hướng dẫn phong cách
# EN: Benefit: Pre-training diffusion model improves style guidance quality

# VN: Tải mô hình khuếch tán nếu đã tồn tại
# EN: Load diffusion model if it exists
if os.path.exists('targetdiff.pth'):
    diffusion_model.load_state_dict(torch.load('targetdiff.pth'))  # VN: Tải trạng thái mô hình khuếch tán
                                                                  # EN: Load diffusion model state
diffusion_model.eval()  # VN: Đặt mô hình khuếch tán ở chế độ eval
                        # EN: Set diffusion model to eval mode
# VN: Lợi ích: Tái sử dụng mô hình đã huấn luyện để tiết kiệm thời gian
# EN: Benefit: Reuses pre-trained model to save time

# VN: Khởi tạo trọng số hướng dẫn khuếch tán
# EN: Initialize diffusion guidance weight
diffusion_weight = 0.0  # VN: Bắt đầu với trọng số 0, sẽ tăng dần
                        # EN: Start with weight 0, will increase gradually
# VN: Lợi ích: Dần dần tích hợp hướng dẫn khuếch tán để ổn định huấn luyện
# EN: Benefit: Gradually integrates diffusion guidance for training stability

# VN: Tính số batch nhỏ nhất để xử lý dataset không đồng đều
# EN: Calculate minimum number of batches to handle uneven datasets
min_batches = min(len(dataloader_A), len(dataloader_B))  # VN: Lấy số batch nhỏ nhất giữa hai miền
                                                         # EN: Take minimum batch count between domains
# VN: Lợi ích: Đảm bảo số vòng lặp đồng bộ giữa hai DataLoader
# EN: Benefit: Ensures synchronized iteration across both DataLoaders

# VN: Vòng lặp huấn luyện chính
# EN: Main training loop
for epoch in tqdm(range(opt.n_epochs), desc="Main Training"):
    # VN: Tạo iterator mới cho mỗi epoch
    # EN: Create new iterators for each epoch
    data_iter_A = iter(dataloader_A)  # VN: Iterator cho miền A
                                      # EN: Iterator for domain A
    data_iter_B = iter(dataloader_B)  # VN: Iterator cho miền B
                                      # EN: Iterator for domain B
    
    # VN: Cập nhật trọng số khuếch tán theo chiến lược tăng dần
    # EN: Update diffusion weight with gradual ramp-up
    diffusion_weight = min(1.0, (epoch + 1) / 200)  # VN: Tăng dần trọng số qua 200 epoch
                                                    # EN: Gradually increase weight over 200 epochs
    # VN: Lợi ích: Tích hợp từ từ ảnh khuếch tán để tránh sốc mô hình
    # EN: Benefit: Gradually integrates diffused images to avoid model shock
    
    if np.mean(target_performance) > 0.75:  # VN: Cập nhật trọng số mất mát nếu hiệu suất mục tiêu cao
        lambda_style = min(2.0, lambda_style * 1.05)  # VN: Tăng lambda_style tối đa đến 2.0
        lambda_content = max(0.1, lambda_content * 0.95)  # VN: Giảm lambda_content tối thiểu đến 0.1
        print(f"Curriculum update: λ_style={lambda_style:.3f}, λ_content={lambda_content:.3f}")  # VN: In cập nhật
                                                                                       # EN: Print update
    # VN: Lợi ích: Chiến lược học theo chương trình điều chỉnh trọng số để tối ưu hóa
    # EN: Benefit: Curriculum learning adjusts weights for optimization
    
    for i in tqdm(range(min_batches), desc=f"Epoch {epoch}"):  # VN: Duyệt qua các batch
        # VN: Lấy batch tiếp theo
        # EN: Get next batch
        imgs_A, labels_A = next(data_iter_A)  # VN: Lấy ảnh và nhãn từ miền A
                                              # EN: Get images and labels from domain A
        imgs_B, labels_B = next(data_iter_B)  # VN: Lấy ảnh và nhãn từ miền B
                                              # EN: Get images and labels from domain B
        
        imgs_A = imgs_A.to(device)  # VN: Chuyển ảnh miền A sang thiết bị
                                    # EN: Move domain A images to device
        labels_A = labels_A.to(device)  # VN: Chuyển nhãn miền A sang thiết bị
                                        # EN: Move domain A labels to device
        imgs_B = imgs_B.to(device)  # VN: Chuyển ảnh miền B sang thiết bị
                                    # EN: Move domain B images to device
        labels_B = labels_B.to(device)  # VN: Chuyển nhãn miền B sang thiết bị
                                        # EN: Move domain B labels to device
        
        batch_size = imgs_A.size(0)  # VN: Lấy kích thước batch
                                     # EN: Get batch size
        
        # VN: Tạo ground truth cho huấn luyện đối kháng
        # EN: Create ground truth for adversarial training
        valid = torch.ones(batch_size, 1, requires_grad=False).to(device)  # VN: Tensor 1 cho ảnh thật
                                                                  # EN: Tensor of ones for real images
        fake = torch.zeros(batch_size, 1, requires_grad=False).to(device)  # VN: Tensor 0 cho ảnh giả
                                                                  # EN: Tensor of zeros for fake images
        
        # VN: Tạo ảnh khuếch tán từ miền A
        # EN: Generate diffused images from domain A
        with torch.no_grad():  # VN: Tắt gradient để suy luận
            diffused = sample_conditional_diffusion(
                diffusion_model, 
                imgs_A, 
                opt.channels,
                opt.img_size,
                betas,
                device
            )  # VN: Tạo ảnh khuếch tán từ mô hình khuếch tán
               # EN: Generate diffused images from diffusion model
        
        diffused = diffused.detach()  # VN: Ngắt gradient để sử dụng như đầu vào cố định
                                      # EN: Detach gradient to use as fixed input
        
        # VN: Kết hợp ảnh gốc và ảnh khuếch tán
        # EN: Blend source and diffused images
        source_imgs = (1 - diffusion_weight) * imgs_A + diffusion_weight * diffused  # VN: Kết hợp theo tỷ lệ diffusion_weight
                                                                            # EN: Blend based on diffusion_weight
        source_labels = labels_A  # VN: Giữ nhãn gốc của miền A
                                  # EN: Keep original labels from domain A
        # VN: Lợi ích: Kết hợp ảnh khuếch tán giúp hướng dẫn phong cách miền mục tiêu
        # EN: Benefit: Blending diffused images guides style towards target domain
        
        # VN: Tạo nhiễu ngẫu nhiên
        # EN: Sample random noise
        z = torch.randn(batch_size, opt.latent_dim).to(device)  # VN: Tạo tensor nhiễu ngẫu nhiên
                                                        # EN: Generate random noise tensor
        
        # =============================================
        # TRAIN GENERATOR
        # =============================================
        optimizer_G.zero_grad()  # VN: Xóa gradient cũ của Generator và Classifier
                                 # EN: Clear old gradients for Generator and Classifier
        
        # VN: Tạo ảnh giống miền mục tiêu
        # EN: Generate target-like images
        fake_B = generator(source_imgs, z)  # VN: Sinh ảnh từ ảnh nguồn và nhiễu
                                            # EN: Generate images from source images and noise
        
        # VN: Phân loại ảnh sinh ra
        # EN: Classify generated images
        label_pred_B = classifier(fake_B)  # VN: Dự đoán nhãn cho ảnh sinh ra
                                           # EN: Predict labels for generated images
        
        # VN: Tính mất mát phân loại
        # EN: Calculate classification loss
        ce_loss = task_loss(label_pred_B, labels_A)  # VN: Mất mát CrossEntropy giữa nhãn dự đoán và nhãn gốc
                                                     # EN: CrossEntropy loss between predicted and original labels
        
        # VN: Tính mất mát nhất quán ngữ nghĩa
        # EN: Calculate semantic consistency loss
        with torch.no_grad():
            real_logits = classifier(imgs_B)  # VN: Lấy logits của ảnh thật từ miền B
                                             # EN: Get logits of real images from domain B
        semantic_loss = semantic_consistency_loss(real_logits, label_pred_B)  # VN: Tính KL divergence để đảm bảo nhất quán
                                                                     # EN: Compute KL divergence for consistency
        
        # VN: Tính mất mát bảo toàn nội dung
        # EN: Calculate content preservation loss
        content_loss = semantic_module.content_loss(source_imgs, fake_B)  # VN: Tính mất mát nội dung giữa ảnh nguồn và ảnh sinh
                                                                 # EN: Compute content loss between source and generated images
        
        # VN: Hướng dẫn phong cách từ mô hình khuếch tán
        # EN: Style guidance from diffusion model
        with torch.no_grad():
            real_style = style_anchor(imgs_B)  # VN: Trích xuất đặc trưng phong cách từ ảnh thật
                                               # EN: Extract style features from real images
        fake_style = style_anchor(fake_B)  # VN: Trích xuất đặc trưng phong cách từ ảnh sinh
                                           # EN: Extract style features from generated images
        
        style_loss = 0
        for f_real, f_fake in zip(real_style, fake_style):  # VN: So sánh đặc trưng phong cách giữa ảnh thật và giả
            style_loss += F.l1_loss(f_real.detach(), f_fake)  # VN: Tính mất mát L1 cho từng cặp đặc trưng
                                                             # EN: Compute L1 loss for each feature pair
        style_loss /= len(real_style)  # VN: Trung bình mất mát phong cách
                                       # EN: Average style loss
        # VN: Lợi ích: Đảm bảo ảnh sinh có phong cách giống miền mục tiêu
        # EN: Benefit: Ensures generated images match target domain style
        
        # VN: Tính mất mát phân biệt miền
        # EN: Calculate domain confusion loss
        domain_pred = discriminator(fake_B)  # VN: Dự đoán miền từ Discriminator
                                             # EN: Predict domain from Discriminator
        domain_loss = adversarial_loss(domain_pred, valid)  # VN: Mất mát MSE để đánh lừa Discriminator
                                                           # EN: MSE loss to fool Discriminator
        
        # VN: Tính mất mát LPIPS (nhận thức)
        # EN: Calculate LPIPS (perceptual) loss
        lpips_loss = loss_fn(preprocess(fake_B), preprocess(imgs_B)).mean()  # VN: Tính mất mát LPIPS để đo độ giống nhận thức
                                                                     # EN: Compute LPIPS loss for perceptual similarity
        
        # VN: Tổng mất mát Generator
        # EN: Total Generator loss
        g_loss = (lambda_adv * domain_loss + 
                lambda_task * ce_loss + 
                lambda_semantic * semantic_loss + 
                lambda_style * style_loss +
                lambda_content * content_loss +
                lpips_loss)  # VN: Kết hợp tất cả các mất mát với trọng số tương ứng
                             # EN: Combine all losses with corresponding weights
        
        g_loss.backward()  # VN: Tính gradient
                           # EN: Compute gradients
        # VN: Cắt gradient để ổn định
        # EN: Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)  # VN: Giới hạn gradient Generator
                                                                      # EN: Limit Generator gradient norm
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)  # VN: Giới hạn gradient Classifier
                                                                      # EN: Limit Classifier gradient norm
        optimizer_G.step()  # VN: Cập nhật trọng số Generator và Classifier
                            # EN: Update Generator and Classifier weights
        ema_g.update()  # VN: Cập nhật trọng số shadow của EMA
                        # EN: Update EMA shadow weights
        # VN: Lợi ích: Tối ưu hóa Generator để tạo ảnh giống miền mục tiêu và giữ nội dung
        # EN: Benefit: Optimizes Generator to produce target-like images while preserving content
        
        # =============================================
        # TRAIN DISCRIMINATOR
        # =============================================
        if step % opt.n_critic == 0:  # VN: Huấn luyện Discriminator sau mỗi n_critic bước
            optimizer_D.zero_grad()  # VN: Xóa gradient cũ
                                     # EN: Clear old gradients
            
            # VN: Dự đoán trên ảnh thật
            # EN: Predict on real images
            pred_real = discriminator(imgs_B)  # VN: Dự đoán miền cho ảnh thật
                                               # EN: Predict domain for real images
            real_loss = adversarial_loss(pred_real, valid)  # VN: Mất mát MSE cho ảnh thật
                                                           # EN: MSE loss for real images
            
            # VN: Dự đoán trên ảnh giả
            # EN: Predict on fake images
            pred_fake = discriminator(fake_B.detach())  # VN: Dự đoán miền cho ảnh giả (ngắt gradient)
                                                       # EN: Predict domain for fake images (detached)
            fake_loss = adversarial_loss(pred_fake, fake)  # VN: Mất mát MSE cho ảnh giả
                                                          # EN: MSE loss for fake images
            
            d_loss = (real_loss + fake_loss) / 2  # VN: Tổng mất mát Discriminator (trung bình)
                                                  # EN: Total Discriminator loss (averaged)
            d_loss.backward()  # VN: Tính gradient
                               # EN: Compute gradients
            optimizer_D.step()  # VN: Cập nhật trọng số Discriminator
                                # EN: Update Discriminator weights
        # VN: Lợi ích: Huấn luyện Discriminator để phân biệt ảnh thật và giả, cải thiện Generator
        # EN: Benefit: Trains Discriminator to distinguish real and fake images, improving Generator
        
        step += 1  # VN: Tăng bộ đếm bước
                   # EN: Increment step counter
        
        # =============================================
        # EVALUATION 
        # =============================================
        # VN: Đánh giá hiệu suất
        # EN: Evaluate performance
        gen_preds = torch.argmax(label_pred_B, dim=1)  # VN: Lấy nhãn dự đoán từ ảnh sinh
                                                       # EN: Get predicted labels from generated images
        correct_gen = (gen_preds == labels_A).float().mean()  # VN: Tính độ chính xác trên ảnh sinh
                                                              # EN: Calculate accuracy on generated images
        task_performance.append(correct_gen.item())  # VN: Lưu độ chính xác
                                                    # EN: Store accuracy
        if len(task_performance) > 100:  # VN: Giới hạn danh sách ở 100 giá trị
            task_performance.pop(0)  # VN: Xóa giá trị cũ nhất
                                     # EN: Remove oldest value
        
        with torch.no_grad():
            target_preds = classifier(imgs_B)  # VN: Dự đoán nhãn cho ảnh mục tiêu
                                              # EN: Predict labels for target images
        target_pred_labels = torch.argmax(target_preds, dim=1)  # VN: Lấy nhãn dự đoán
                                                               # EN: Get predicted labels
        correct_target = (target_pred_labels == labels_B).float().mean()  # VN: Tính độ chính xác trên ảnh mục tiêu
                                                                  # EN: Calculate accuracy on target images
        target_performance.append(correct_target.item())  # VN: Lưu độ chính xác
                                                         # EN: Store accuracy
        if len(target_performance) > 100:  # VN: Giới hạn danh sách ở 100 giá trị
            target_performance.pop(0)  # VN: Xóa giá trị cũ nhất
                                       # EN: Remove oldest value
        
        if correct_target.item() >= best:  # VN: Lưu mô hình nếu độ chính xác mục tiêu tốt hơn
            best = correct_target.item()  # VN: Cập nhật độ chính xác tốt nhất
                                          # EN: Update best accuracy
            ema_g.apply_shadow()  # VN: Áp dụng trọng số shadow của EMA
                                  # EN: Apply EMA shadow weights
            torch.save(generator.state_dict(), 'best_generator.pth')  # VN: Lưu trạng thái Generator
                                                             # EN: Save Generator state
            ema_g.restore()  # VN: Khôi phục trọng số gốc
                             # EN: Restore original weights
            print(f"Saved model at epoch {epoch} with target acc = {best:.4f}")  # VN: In thông tin lưu mô hình
                                                                         # EN: Print model save information
        
        # VN: Ghi log huấn luyện
        # EN: Log training information
        log_entry = {
            "epoch": epoch,
            "batch": i,
            "g_loss": g_loss.item(),
            "d_loss": d_loss.item() if step % opt.n_critic == 0 else 0,
            "ce_loss": ce_loss.item(),
            "semantic_loss": semantic_loss.item(),
            "content_loss": content_loss.item(),
            "style_loss": style_loss.item(),
            "domain_loss": domain_loss.item(),
            "lpips_loss": lpips_loss.item(),
            "acc": correct_gen.item(),
            "target_acc": correct_target.item(),
            "avg_acc": np.mean(task_performance),
            "avg_target_acc": np.mean(target_performance),
            "lambda_style": lambda_style,
            "lambda_content": lambda_content,
            "diffusion_weight": diffusion_weight
        }  # VN: Tạo bản ghi log với các thông tin quan trọng
           # EN: Create log entry with key training information
        log_data.append(log_entry)  # VN: Thêm bản ghi vào danh sách log
                                    # EN: Add entry to log list
        
        print(
            f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{min_batches}] "
            f"[D loss: {d_loss.item() if step % opt.n_critic == 0 else 0:.4f}] "
            f"[G loss: {g_loss.item():.4f}] "
            f"[Acc: {100*correct_gen.item():.1f}% ({100*np.mean(task_performance):.1f}%)] "
            f"[Target: {100*correct_target.item():.1f}% ({100*np.mean(target_performance):.1f}%)] "
            f"[Diff Weight: {diffusion_weight:.2f}]"
        )  # VN: In thông tin huấn luyện mỗi batch
           # EN: Print training information per batch
        
        # VN: Lưu ảnh mẫu
        # EN: Save sample images
        if step % opt.sample_interval == 0:  # VN: Lưu ảnh sau mỗi khoảng sample_interval
            with torch.no_grad():
                sample_imgs = torch.cat((
                    imgs_A.data[:5], 
                    diffused.data[:5], 
                    fake_B.data[:5], 
                    imgs_B.data[:5]
                ), -1)  # VN: Nối ảnh nguồn, khuếch tán, sinh ra và thật để so sánh
                        # EN: Concatenate source, diffused, generated, and real images for comparison
                save_image(
                    sample_imgs, 
                    f"imagesss/sample_{epoch}_{i}.png", 
                    nrow=5, 
                    normalize=True
                )  # VN: Lưu ảnh vào thư mục imagesss
                   # EN: Save images to imagesss directory
            # VN: Lợi ích: Giúp theo dõi trực quan chất lượng ảnh sinh ra
            # EN: Benefit: Enables visual tracking of generated image quality
    
    # VN: Lưu checkpoint sau mỗi 50 epoch
    # EN: Save checkpoint every 50 epochs
    if epoch % 50 == 0:
        ema_g.apply_shadow()  # VN: Áp dụng trọng số shadow
                              # EN: Apply shadow weights
        torch.save({
            'generator': generator.state_dict(),
            'classifier': classifier.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
            'epoch': epoch,
            'best_acc': best
        }, f'checkpoint/checkpoint_{epoch}.pth')  # VN: Lưu trạng thái các mô hình và tối ưu hóa
                                                 # EN: Save model and optimizer states
        ema_g.restore()  # VN: Khôi phục trọng số gốc
                         # EN: Restore original weights
    # VN: Lợi ích: Lưu checkpoint giúp khôi phục huấn luyện nếu bị gián đoạn
    # EN: Benefit: Checkpoints enable resuming training if interrupted
    
    # VN: Cập nhật scheduler
    # EN: Update schedulers
    scheduler_G.step()  # VN: Cập nhật tốc độ học cho Generator
                        # EN: Update learning rate for Generator
    scheduler_D.step()  # VN: Cập nhật tốc độ học cho Discriminator
                        # EN: Update learning rate for Discriminator
    
    # VN: Lưu log huấn luyện
    # EN: Save training logs
    with open("training_log.json", "w") as f:
        json.dump(log_data, f, indent=2)  # VN: Ghi log vào file JSON
                                          # EN: Write logs to JSON file
    # VN: Lợi ích: Lưu log giúp phân tích và theo dõi tiến trình huấn luyện
    # EN: Benefit: Logging enables analysis and tracking of training progress

# VN: Lưu mô hình cuối cùng
# EN: Save final model
ema_g.apply_shadow()  # VN: Áp dụng trọng số shadow cho Generator
                      # EN: Apply shadow weights to Generator
torch.save(generator.state_dict(), 'final_generator.pth')  # VN: Lưu trạng thái Generator cuối cùng
                                                          # EN: Save final Generator state
ema_g.restore()  # VN: Khôi phục trọng số gốc
                 # EN: Restore original weights
print("Training completed!")  # VN: Thông báo hoàn thành huấn luyện
                              # EN: Print training completion
# VN: Lợi ích: Lưu mô hình cuối cùng để sử dụng trong suy luận
# EN: Benefit: Saves final model for use in inference