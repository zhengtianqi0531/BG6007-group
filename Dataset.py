import pandas as pd  # 用于读取和处理 CSV 文件
import numpy as np   # 用于数值计算和数组操作
import torch         # PyTorch 深度学习框架
from torch.utils.data import Dataset  # PyTorch 数据集基类

class OximetryDataset(Dataset):
    def __init__(self, ppg_file, gt_file, window_size=300):
        self.ppg_df = pd.read_csv(ppg_file)  # Remove index_col=0
        self.gt_df = pd.read_csv(gt_file)  # Already removed

        self.data = pd.merge(
            self.ppg_df,
            self.gt_df,
            left_index=True,
            right_index=True,
            how='inner'
        )

        print("[DEBUG] Columns after merge:")
        print(self.data.columns.tolist())
        print()

        self.window_size = window_size
        ppg_columns = ['R', 'G', 'B']

        self.data[ppg_columns] = (
                                         self.data[ppg_columns] - self.data[ppg_columns].mean()
                                 ) / self.data[ppg_columns].std()

        self.ppg_columns = ppg_columns

        if 'SpO2_gt' in self.data.columns:
            self.spo2_column = 'SpO2_gt'
        elif 'sp02_1' in self.data.columns:
            self.spo2_column = 'sp02_1'  # 使用第一个脉搏血氧仪的读数
        else:
            # 如果找不到明确的列，尝试找包含 'spo2' 或 'sp02' 的列
            spo2_cols = [col for col in self.data.columns
                         if 'spo2' in col.lower() or 'sp02' in col.lower()]
            if spo2_cols:
                self.spo2_column = spo2_cols[0]
            else:
                raise ValueError("SpO2 true value column not found! Please check the data file.")

        print(f"Dataset loading completed!")
        print(f"   - PPG file: {ppg_file}")
        print(f"   - GT file: {gt_file}")
        print(f"   - Total number of data rows: {len(self.data)}")
        print(f"   - window size: {window_size}")
        print(f"   - SpO2 colum: {self.spo2_column}")
        print(f"   - Number of samples that can be generated: {len(self)}")

    def __len__(self):
        return len(self.data) - self.window_size + 1

    def __getitem__(self, idx):
        ppg_window = self.data[self.ppg_columns].iloc[
                     idx: idx + self.window_size].values

        spo2_gt = self.data[self.spo2_column].iloc[idx + self.window_size - 1]

        classification_target = 1.0 if spo2_gt < 90 else 0.0
        ppg_tensor = torch.tensor(ppg_window, dtype=torch.float32)

        ppg_tensor = ppg_tensor.permute(1, 0)

        regression_target = torch.tensor(spo2_gt, dtype=torch.float32)

        classification_target = torch.tensor(
            classification_target,
            dtype=torch.float32
        )

        return ppg_tensor, regression_target, classification_target


# THIS SHOULD BE AT TOP LEVEL - NO INDENTATION!
if __name__ == "__main__":
    """
    这是一个测试代码，演示如何使用这个数据集类
    """
    import os

    print("=" * 70)
    print("Blood oxygen dataset testing")
    print("=" * 70)

    # DEBUG: Check current directory and files
    print("\n[DEBUG] Current working directory:")
    print(os.getcwd())
    print("\n[DEBUG] Files in current directory:")
    for file in os.listdir('.'):
        print(f"  - {file}")
    print()

    ppg_file = "100002.csv"
    gt_file = "100002 (1).csv"

    # DEBUG: Check if files exist
    print(f"[DEBUG] Does {ppg_file} exist? {os.path.exists(ppg_file)}")
    print(f"[DEBUG] Does {gt_file} exist? {os.path.exists(gt_file)}")
    print()

    try:
        dataset = OximetryDataset(
            ppg_file=ppg_file,
            gt_file=gt_file,
            window_size=300
        )
        print(f"\nDataset loading completed!")

    except FileNotFoundError:
        print(f"\nFile not found! Please check the path：")
        print(f"   PPG file: {ppg_file}")
        print(f"   GT file: {gt_file}")
        exit()

    except Exception as e:
        print(f"\nError creating dataset：{e}")
        exit()

    print(f"\nDataset information:")
    print(f"   - Total sample size: {len(dataset)}")
    print(f"   - Window Size: {dataset.window_size} Hz")
    print(f"   - Data duration: {dataset.window_size / 30:.1f} s")

    ppg, reg_target, cls_target = dataset[0]

    print(f"\nFirst Sample:")
    print(f"   - PPG tensor shape: {ppg.shape}")
    print(f"   - PPG data range: [{ppg.min():.3f}, {ppg.max():.3f}]")
    print(f"   - Return to Goal（SpO2）: {reg_target.item():.2f}%")
    print(f"   - Classification target (Hypoxia): {cls_target.item()}")
    print(f"     → Explanation: {'Risk of Hypoxia' if cls_target.item() == 1.0 else 'Normal'}")

    from torch.utils.data import DataLoader

    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=16,  # 每批 16 个样本
        shuffle=True,  # 随机打乱
        num_workers=0  # 单进程加载（Windows 上建议设为0）
    )

    print(f"\n使用 DataLoader:")
    print(f"   - Batch size: 16")
    print(f"   - Total number of batches: {len(dataloader)}")

    # 获取第一个批次
    batch_ppg, batch_reg, batch_cls = next(iter(dataloader))

    print(f"\nthe first batch:")
    print(f"   - PPG Batch shape: {batch_ppg.shape}")
    print(f"   - Return to the target shape: {batch_reg.shape}")
    print(f"   - Classification target shape: {batch_cls.shape}")

    hypoxemia_count = 0
    normal_count = 0

    for i in range(min(1000, len(dataset))):  # 检查前1000个样本
        _, _, cls_label = dataset[i]
        if cls_label.item() == 1.0:
            hypoxemia_count += 1
        else:
            normal_count += 1

    total = hypoxemia_count + normal_count
    print(f"\nDataset statistics（first{total}samples）:")
    print(f"   - Hypoxic Sample: {hypoxemia_count} ({hypoxemia_count / total * 100:.1f}%)")
    print(f"   - Normal Sample: {normal_count} ({normal_count / total * 100:.1f}%)")

    if hypoxemia_count / total < 0.1 or hypoxemia_count / total > 0.9:
        print(f"   Warning: Data imbalance! Suggest using a weighted loss function")

    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)

import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import time


# ============================================
# YOUR MODEL DEFINITION (Keep this the same)
# ============================================
class HybridOximetryModel(nn.Module):
    def __init__(self, input_channels, window_size=300):
        super(HybridOximetryModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()

        # Calculate size after convolutions and pooling
        conv_output_size = 64 * 75  # 64 channels * 75 time points

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, 64)

        # Output heads
        self.regression_head = nn.Linear(64, 1)
        self.classification_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Convolutional feature extraction
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        # Two output heads
        reg_output = self.regression_head(x)
        cls_output = self.classification_head(x)

        return reg_output, cls_output


# ============================================
# FINGER DETECTOR
# ============================================
class FingerDetector:
    """使用颜色检测来定位手指"""

    def __init__(self):
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    def detect_finger(self, frame):
        """检测手指区域"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) < 1000:
            return None

        x, y, w, h = cv2.boundingRect(largest_contour)

        margin = 0.2
        x1 = int(x + w * margin)
        y1 = int(y + h * margin)
        x2 = int(x + w * (1 - margin))
        y2 = int(y + h * (1 - margin))

        return (x1, y1, x2, y2)


# ============================================
# PPG EXTRACTOR
# ============================================
class AdvancedPPGExtractor:
    """高级 PPG 信号提取器"""

    def __init__(self, buffer_size=300, fps=30):
        self.buffer_size = buffer_size
        self.fps = fps
        self.red_buffer = deque(maxlen=buffer_size)
        self.green_buffer = deque(maxlen=buffer_size)
        self.blue_buffer = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)
        self.finger_detector = FingerDetector()

    def extract_roi(self, frame, use_finger_detection=True):
        """提取 ROI"""
        if use_finger_detection:
            roi_coords = self.finger_detector.detect_finger(frame)
            if roi_coords is None:
                height, width = frame.shape[:2]
                roi_size = min(width, height) // 3
                center_x, center_y = width // 2, height // 2
                x1 = center_x - roi_size // 2
                y1 = center_y - roi_size // 2
                x2 = center_x + roi_size // 2
                y2 = center_y + roi_size // 2
                roi_coords = (x1, y1, x2, y2)
        else:
            height, width = frame.shape[:2]
            roi_size = min(width, height) // 3
            center_x, center_y = width // 2, height // 2
            x1 = center_x - roi_size // 2
            y1 = center_y - roi_size // 2
            x2 = center_x + roi_size // 2
            y2 = center_y + roi_size // 2
            roi_coords = (x1, y1, x2, y2)

        x1, y1, x2, y2 = roi_coords
        roi = frame[y1:y2, x1:x2]

        return roi, roi_coords

    def add_frame(self, frame, use_finger_detection=True):
        """添加新帧"""
        roi, roi_coords = self.extract_roi(frame, use_finger_detection)

        b_mean = np.mean(roi[:, :, 0])
        g_mean = np.mean(roi[:, :, 1])
        r_mean = np.mean(roi[:, :, 2])

        self.blue_buffer.append(b_mean)
        self.green_buffer.append(g_mean)
        self.red_buffer.append(r_mean)
        self.timestamps.append(time.time())

        return roi_coords

    def get_signals(self):
        """获取信号"""
        if len(self.red_buffer) < self.buffer_size:
            return None, None, None

        r = np.array(self.red_buffer)
        g = np.array(self.green_buffer)
        b = np.array(self.blue_buffer)

        return r, g, b

    def assess_signal_quality(self):
        """评估信号质量"""
        r, g, b = self.get_signals()
        if r is None:
            return 0.0

        cv_g = np.std(g) / (np.mean(g) + 1e-8)

        quality = 0.0
        if 0.01 <= cv_g <= 0.1:
            quality = 1.0 - abs(cv_g - 0.05) / 0.05

        return np.clip(quality, 0.0, 1.0)

    def is_ready(self):
        """检查是否准备好"""
        return len(self.red_buffer) >= self.buffer_size


# ============================================
# SPO2 MONITOR (MODIFIED FOR YOUR MODEL)
# ============================================
class IntegratedSpO2Monitor:
    """集成您训练好的模型的血氧监测器"""

    def __init__(self, model_path=None, window_size=300, use_finger_detection=True, device='cpu'):
        self.window_size = window_size
        self.use_finger_detection = use_finger_detection
        self.device = device
        self.ppg_extractor = AdvancedPPGExtractor(buffer_size=window_size)

        # Load your trained model
        self.model = None
        if model_path:
            self.load_model(model_path)

        # For smoothing SpO2 readings
        self.spo2_history = deque(maxlen=10)
        self.hypoxemia_history = deque(maxlen=10)

    def load_model(self, model_path):
        """加载您训练好的模型"""
        try:
            print(f"Loading model from {model_path}...")
            checkpoint = torch.load(model_path, map_location=self.device)

            # Create model instance (3 channels: R, G, B)
            self.model = HybridOximetryModel(
                input_channels=3,
                window_size=self.window_size
            )

            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            print(f"✓ Model loaded successfully!")
            print(f"  Device: {self.device}")

        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("  Will use simple estimation method instead.")
            self.model = None

    def preprocess_signals(self, r, g, b):
        """
        预处理信号 - 必须与训练时相同！
        """
        signals = np.stack([r, g, b], axis=1)

        # Z-score 标准化（与 Dataset.py 中相同）
        mean = signals.mean(axis=0)
        std = signals.std(axis=0)
        normalized = (signals - mean) / (std + 1e-8)

        return normalized

    def predict_spo2(self):
        """
        预测 SpO2 - 适配您的双输出模型
        """
        r, g, b = self.ppg_extractor.get_signals()

        if r is None:
            return None, None

        # 预处理
        normalized = self.preprocess_signals(r, g, b)

        if self.model is None:
            # 简单估计方法
            r_ac = np.std(r)
            r_dc = np.mean(r)
            ir_ac = np.std(b)
            ir_dc = np.mean(b)

            ratio = (r_ac / r_dc) / (ir_ac / ir_dc + 1e-8)
            spo2 = 110 - 25 * ratio
            spo2 = np.clip(spo2, 70, 100)
            hypoxemia_risk = 1.0 if spo2 < 90 else 0.0
        else:
            # 使用您的模型预测
            with torch.no_grad():
                # 转换为 tensor (1, 3, 300)
                ppg_tensor = torch.tensor(normalized.T, dtype=torch.float32)
                ppg_tensor = ppg_tensor.unsqueeze(0).to(self.device)

                # 预测 - 您的模型返回两个输出
                reg_output, cls_output = self.model(ppg_tensor)

                # 提取值
                spo2 = reg_output.squeeze().item()
                hypoxemia_risk = cls_output.squeeze().item()

                # 限制 SpO2 范围
                spo2 = np.clip(spo2, 70, 100)

        # 平滑读数
        self.spo2_history.append(spo2)
        self.hypoxemia_history.append(hypoxemia_risk)

        spo2_smoothed = np.median(self.spo2_history)
        hypoxemia_smoothed = np.mean(self.hypoxemia_history)

        return spo2_smoothed, hypoxemia_smoothed

    def draw_signal_plot(self, frame, signals):
        """在画面上绘制信号波形"""
        if signals is None:
            return

        r, g, b = signals
        height, width = frame.shape[:2]

        plot_height = 100
        plot_width = 300
        plot_x = width - plot_width - 10
        plot_y = 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (plot_x, plot_y),
                      (plot_x + plot_width, plot_y + plot_height),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        display_length = min(150, len(g))
        g_display = g[-display_length:]

        g_norm = (g_display - g_display.min()) / (g_display.max() - g_display.min() + 1e-8)
        g_norm = g_norm * (plot_height - 20) + 10

        points = []
        for i, val in enumerate(g_norm):
            x = plot_x + int((i / display_length) * plot_width)
            y = plot_y + plot_height - int(val)
            points.append((x, y))

        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], (0, 255, 0), 2)

        cv2.putText(frame, "PPG Signal", (plot_x, plot_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def run(self, camera_id=0):
        """运行监测"""
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print("错误：无法打开摄像头")
            return

        print("=" * 70)
        print("实时血氧监测（使用您的训练模型）")
        print("=" * 70)
        print("\n提示:")
        if self.use_finger_detection:
            print("  1. 将手指放在摄像头前（会自动检测）")
        else:
            print("  1. 将手指放在绿色框内")
        print("  2. 保持手指稳定，确保光线充足")
        print("  3. 等待信号质量达到良好水平")
        print("  4. 按 'q' 退出，按 's' 截图\n")

        if self.model:
            print("✓ 使用训练好的深度学习模型\n")
        else:
            print("⚠ 使用简单估计方法（未加载模型）\n")

        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            roi_coords = self.ppg_extractor.add_frame(frame, self.use_finger_detection)
            x1, y1, x2, y2 = roi_coords

            quality = self.ppg_extractor.assess_signal_quality()

            if self.ppg_extractor.is_ready() and quality > 0.5:
                color = (0, 255, 0)
                status = "信号良好"
            elif self.ppg_extractor.is_ready():
                color = (0, 165, 255)
                status = "信号质量差"
            else:
                color = (0, 165, 255)
                progress = len(self.ppg_extractor.red_buffer)
                status = f"采集中... {progress}/{self.window_size}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            quality_text = f"Quality: {quality * 100:.0f}%"
            cv2.putText(frame, quality_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if self.ppg_extractor.is_ready() and quality > 0.3:
                spo2, hypoxemia_risk = self.predict_spo2()

                if spo2 is not None:
                    spo2_text = f"SpO2: {spo2:.1f}%"

                    if spo2 >= 95:
                        text_color = (0, 255, 0)
                        health_status = "Normal"
                    elif spo2 >= 90:
                        text_color = (0, 255, 255)
                        health_status = "Low"
                    else:
                        text_color = (0, 0, 255)
                        health_status = "Critical"

                    cv2.putText(frame, spo2_text, (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)
                    cv2.putText(frame, health_status, (10, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

                    # 显示低氧风险（来自分类头）
                    risk_text = f"Hypoxemia Risk: {hypoxemia_risk * 100:.0f}%"
                    risk_color = (0, 0, 255) if hypoxemia_risk > 0.5 else (0, 255, 0)
                    cv2.putText(frame, risk_text, (10, 170),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, risk_color, 2)

                    signals = self.ppg_extractor.get_signals()
                    self.draw_signal_plot(frame, signals)

            fps_counter += 1
            if time.time() - fps_start_time > 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()

            cv2.putText(frame, f"FPS: {current_fps}",
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.putText(frame, "Press 'q':quit 's':screenshot",
                        (frame.shape[1] - 300, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('SpO2 Monitor with Your Model', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"spo2_screenshot_{int(time.time())}.png"
                cv2.imwrite(filename, frame)
                print(f"截图已保存: {filename}")

        cap.release()
        cv2.destroyAllWindows()
        print("\n监测结束")


# ============================================
# MAIN PROGRAM
# ============================================
if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建监测器并加载您的模型
    monitor = IntegratedSpO2Monitor(
        model_path='oximetry_model_trained.pth',  # ← 您的模型路径
        window_size=300,
        use_finger_detection=True,
        device=device
    )

    # 运行监测
    monitor.run(camera_id=0)