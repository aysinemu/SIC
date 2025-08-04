# VN: Định nghĩa lớp EMA để thực hiện Exponential Moving Average cho các tham số mô hình
# EN: Define EMA class to implement Exponential Moving Average for model parameters
class EMA:
    # VN: Hàm khởi tạo với mô hình và hệ số decay
    # EN: Constructor with model and decay factor
    def __init__(self, model, decay=0.999):
        self.model = model  # VN: Lưu tham chiếu đến mô hình cần áp dụng EMA
                           # EN: Store reference to the model for EMA
        self.decay = decay  # VN: Lưu hệ số decay (mặc định 0.999) để cân bằng giữa tham số hiện tại và trung bình
                            # EN: Store decay factor (default 0.999) to balance current and averaged parameters
        # VN: Lợi ích: Decay cao (gần 1) giúp giữ lại nhiều thông tin từ các bước trước, ổn định huấn luyện
        # EN: Benefit: High decay (close to 1) retains more information from previous steps, stabilizing training
        self.shadow = {}  # VN: Từ điển lưu các tham số trung bình động (shadow parameters)
                          # EN: Dictionary to store moving average parameters (shadow parameters)
        self.backup = {}  # VN: Từ điển lưu bản sao tham số gốc khi áp dụng shadow
                          # EN: Dictionary to store backup of original parameters when applying shadow
        # VN: Lợi ích: Shadow và backup cho phép quản lý trạng thái tham số dễ dàng
        # EN: Benefit: Shadow and backup enable easy management of parameter states

    # VN: Hàm register, khởi tạo shadow parameters từ tham số mô hình
    # EN: register function, initializes shadow parameters from model parameters
    def register(self):
        for name, param in self.model.named_parameters():  # VN: Duyệt qua các tham số của mô hình
                                                          # EN: Iterate through model's named parameters
            if param.requires_grad:  # VN: Chỉ xử lý các tham số có thể huấn luyện (requires_grad=True)
                                     # EN: Only process trainable parameters (requires_grad=True)
                self.shadow[name] = param.data.clone()  # VN: Sao chép tham số vào shadow để lưu trung bình động
                                                       # EN: Clone parameter data to shadow for moving average
        # VN: Lợi ích: Khởi tạo shadow parameters để theo dõi trung bình động, không ảnh hưởng tham số gốc
        # EN: Benefit: Initializes shadow parameters for tracking moving average without affecting original parameters

    # VN: Hàm update, cập nhật shadow parameters bằng công thức EMA
    # EN: update function, updates shadow parameters using EMA formula
    def update(self):
        for name, param in self.model.named_parameters():  # VN: Duyệt qua các tham số của mô hình
                                                          # EN: Iterate through model's named parameters
            if param.requires_grad:  # VN: Chỉ cập nhật tham số có thể huấn luyện
                                     # EN: Only update trainable parameters
                assert name in self.shadow  # VN: Kiểm tra xem tham số đã được đăng ký trong shadow
                                            # EN: Ensure parameter is registered in shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]  # VN: Tính trung bình động: (1-decay)*tham_số_mới + decay*shadow
                                                                                        # EN: Compute moving average: (1-decay)*new_param + decay*shadow
                self.shadow[name] = new_average.clone()  # VN: Cập nhật shadow với giá trị mới, dùng clone để tránh thay đổi trực tiếp
                                                        # EN: Update shadow with new value, using clone to avoid direct modification
        # VN: Lợi ích: Cập nhật shadow parameters liên tục để duy trì trung bình động, cải thiện độ ổn định của mô hình
        # EN: Benefit: Continuously updates shadow parameters to maintain moving average, improving model stability

    # VN: Hàm apply_shadow, thay thế tham số mô hình bằng shadow parameters
    # EN: apply_shadow function, replaces model parameters with shadow parameters
    def apply_shadow(self):
        for name, param in self.model.named_parameters():  # VN: Duyệt qua các tham số của mô hình
                                                          # EN: Iterate through model's named parameters
            if param.requires_grad:  # VN: Chỉ xử lý tham số có thể huấn luyện
                                     # EN: Only process trainable parameters
                assert name in self.shadow  # VN: Kiểm tra xem shadow parameter tồn tại
                                            # EN: Ensure shadow parameter exists
                self.backup[name] = param.data  # VN: Sao lưu tham số gốc vào backup
                                                # EN: Backup original parameter to backup
                param.data = self.shadow[name]  # VN: Thay thế tham số mô hình bằng shadow parameter
                                                # EN: Replace model parameter with shadow parameter
        # VN: Lợi ích: Cho phép sử dụng shadow parameters trong suy luận, thường cho kết quả ổn định hơn
        # EN: Benefit: Allows using shadow parameters for inference, typically yielding more stable results

    # VN: Hàm restore, khôi phục tham số gốc từ backup
    # EN: restore function, restores original parameters from backup
    def restore(self):
        for name, param in self.model.named_parameters():  # VN: Duyệt qua các tham số của mô hình
                                                          # EN: Iterate through model's named parameters
            if param.requires_grad:  # VN: Chỉ xử lý tham số có thể huấn luyện
                                     # EN: Only process trainable parameters
                assert name in self.backup  # VN: Kiểm tra xem backup parameter tồn tại
                                            # EN: Ensure backup parameter exists
                param.data = self.backup[name]  # VN: Khôi phục tham số gốc từ backup
                                                # EN: Restore original parameter from backup
        self.backup = {}  # VN: Xóa backup sau khi khôi phục để tiết kiệm bộ nhớ
                          # EN: Clear backup after restoration to save memory
        # VN: Lợi ích: Cho phép quay lại trạng thái tham số gốc sau khi dùng shadow, đảm bảo tính linh hoạt
        # EN: Benefit: Allows reverting to original parameters after using shadow, ensuring flexibility