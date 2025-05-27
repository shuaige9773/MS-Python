import os
import cv2
import numpy as np
import pywt
import tkinter as tk
from tkinter import filedialog, ttk, Scale, HORIZONTAL, Scrollbar, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import platform
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
import glob
import time

# ===================== 深度学习模型定义 =====================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation块，用于波段注意力"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, channels, use_se=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels) if use_se else None
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.se:
            out = self.se(out)
        out += residual
        return self.relu(out)

class MSResNet(nn.Module):
    """多光谱残差网络"""
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_blocks=5):
        super(MSResNet, self).__init__()
        
        # 初始卷积层
        self.conv_in = nn.Conv2d(in_channels, num_features, 3, padding=1)
        
        # 残差块
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features, use_se=True) for _ in range(num_blocks)]
        )
        
        # 输出卷积层
        self.conv_out = nn.Conv2d(num_features, out_channels, 3, padding=1)
    
    def forward(self, x):
        # 残差学习：预测噪声
        out = self.conv_in(x)
        out = self.res_blocks(out)
        noise = self.conv_out(out)
        return x - noise  # 返回去噪后的图像

class SPPBlock(nn.Module):
    """光谱金字塔池化块"""
    def __init__(self, in_channels, pool_sizes=[1, 2, 4]):
        super(SPPBlock, self).__init__()
        self.pool_sizes = pool_sizes
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(size) for size in pool_sizes
        ])
        # 输出通道数 = 输入通道数 * 池化尺寸数量
        out_channels = in_channels * len(pool_sizes)
        self.conv = nn.Conv2d(out_channels, in_channels, 1)
    
    def forward(self, x):
        features = []
        for pool in self.pools:
            pooled = pool(x)
            # 上采样到原始尺寸
            upsampled = nn.functional.interpolate(
                pooled, size=x.shape[2:], mode='bilinear', align_corners=False
            )
            features.append(upsampled)
        
        # 拼接所有尺度的特征
        out = torch.cat(features, dim=1)
        out = self.conv(out)
        return out

class DAE_SPP(nn.Module):
    """基于U-Net的去噪自编码器with SPP"""
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super(DAE_SPP, self).__init__()
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 编码器
        for feature in features:
            self.encoder.append(self._block(in_channels, feature))
            in_channels = feature
        
        # SPP模块
        self.spp = SPPBlock(features[-1])
        
        # 解码器
        for feature in reversed(features[:-1]):
            self.decoder.append(
                nn.ConvTranspose2d(in_channels, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(self._block(feature * 2, feature))
            in_channels = feature
        
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        skip_connections = []
        
        # 编码器路径
        for i, down in enumerate(self.encoder):
            x = down(x)
            skip_connections.append(x)
            if i < len(self.encoder) - 1:
                x = self.pool(x)
        
        # SPP处理
        x = self.spp(x)
        
        # 解码器路径
        skip_connections = skip_connections[:-1][::-1]
        
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            if idx // 2 < len(skip_connections):
                skip_connection = skip_connections[idx // 2]
                x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](x)
        
        return self.final_conv(x)

# ===================== 数据集类定义 =====================

class MultispectralDataset(Dataset):
    """多光谱图像数据集"""
    def __init__(self, clean_images, noise_type='gaussian', noise_level=25):
        self.clean_images = clean_images
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.clean_images)
    
    def __getitem__(self, idx):
        clean_img = self.clean_images[idx]
        
        # 添加噪声
        if self.noise_type == 'gaussian':
            noise = np.random.normal(0, self.noise_level, clean_img.shape).astype(np.float32)
            noisy_img = np.clip(clean_img.astype(np.float32) + noise, 0, 255)
        elif self.noise_type == 'salt_pepper':
            noisy_img = clean_img.copy()
            density = self.noise_level / 100.0
            h, w, c = noisy_img.shape
            salt_mask = np.random.random((h, w)) < density / 2
            pepper_mask = np.random.random((h, w)) < density / 2
            for i in range(c):
                noisy_img[salt_mask, i] = 255
                noisy_img[pepper_mask, i] = 0
        else:
            noisy_img = clean_img.copy()
        
        # 转换为tensor
        clean_tensor = torch.from_numpy(clean_img.transpose(2, 0, 1)).float() / 255.0
        noisy_tensor = torch.from_numpy(noisy_img.transpose(2, 0, 1)).float() / 255.0
        
        return noisy_tensor, clean_tensor

# ===================== 主应用程序类 =====================

class ImageDenoisingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("多光谱图像噪声添加与去噪处理")
        self.root.geometry("1280x800")
        
        # 设置样式
        self.setup_styles()
        
        # 设置中文字体
        self.setup_chinese_font()
        
        # 图像变量
        self.original_img = None
        self.noisy_img = None
        self.denoised_img = None
        self.current_displayed = None
        
        # 图像质量评估指标
        self.psnr_value = tk.StringVar(value="PSNR: N/A")
        self.ssim_value = tk.StringVar(value="SSIM: N/A")
        self.mse_value = tk.StringVar(value="MSE: N/A")
        self.sam_value = tk.StringVar(value="SAM: N/A")
        
        # 线程池用于异步处理
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # 进度条变量
        self.progress = tk.DoubleVar()
        
        # 深度学习模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ms_resnet = None
        self.dae_spp = None
        self.models_loaded = False
        
        self.setup_ui()
        
        # 绑定快捷键
        self.setup_shortcuts()
        
        # 初始化深度学习模型
        self.init_deep_learning_models()
    
    def setup_styles(self):
        """设置应用样式"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 自定义颜色方案
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Processing.TButton', font=('Arial', 10, 'bold'))
    
    def setup_chinese_font(self):
        """根据操作系统设置合适的中文字体"""
        system = platform.system()
        if system == 'Windows':
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
        elif system == 'Darwin':  # macOS
            plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC']
        else:  # Linux
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC']
        
        plt.rcParams['axes.unicode_minus'] = False
    
    def setup_shortcuts(self):
        """设置键盘快捷键"""
        self.root.bind('<Control-o>', lambda e: self.load_image())
        self.root.bind('<Control-s>', lambda e: self.save_image())
        self.root.bind('<Control-q>', lambda e: self.root.quit())
    
    def init_deep_learning_models(self):
        """初始化深度学习模型"""
        try:
            # 创建模型实例
            self.ms_resnet = MSResNet(in_channels=3, out_channels=3).to(self.device)
            self.dae_spp = DAE_SPP(in_channels=3, out_channels=3).to(self.device)
            
            # 尝试加载预训练权重
            if os.path.exists('models/ms_resnet.pth'):
                self.ms_resnet.load_state_dict(torch.load('models/ms_resnet.pth', map_location=self.device))
                self.ms_resnet.eval()
            
            if os.path.exists('models/dae_spp.pth'):
                self.dae_spp.load_state_dict(torch.load('models/dae_spp.pth', map_location=self.device))
                self.dae_spp.eval()
            
            self.models_loaded = True
            self.update_status(f"深度学习模型已加载 (设备: {self.device})")
        except Exception as e:
            self.models_loaded = False
            self.update_status(f"深度学习模型加载失败: {str(e)}")
    
    def setup_ui(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        control_frame = self.create_control_panel(main_frame)
        
        # 右侧图像显示区域
        image_frame = self.create_image_display(main_frame)
        
        # 底部状态栏
        self.create_status_bar()
    
    def create_control_panel(self, parent):
        """创建控制面板"""
        # 控制面板外框架
        control_outer_frame = ttk.Frame(parent)
        control_outer_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
        
        # 添加垂直滚动条
        v_scrollbar = Scrollbar(control_outer_frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 创建画布用于滚动
        control_canvas = tk.Canvas(control_outer_frame, yscrollcommand=v_scrollbar.set, width=300)
        control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 配置滚动条
        v_scrollbar.config(command=control_canvas.yview)
        
        # 创建内部框架用于放置控件
        control_frame_inside = ttk.Frame(control_canvas)
        control_canvas_window = control_canvas.create_window((0, 0), window=control_frame_inside, anchor='nw')
        
        # 设置控制面板的标题
        control_label = ttk.Label(control_frame_inside, text="控制面板", style='Title.TLabel')
        control_label.pack(padx=5, pady=5)
        
        # 文件操作区
        self.create_file_operations(control_frame_inside)
        
        # 噪声控制区
        self.create_noise_controls(control_frame_inside)
        
        # 去噪方法区
        self.create_denoise_controls(control_frame_inside)
        
        # 深度学习控制区
        self.create_deep_learning_controls(control_frame_inside)
        
        # 图像质量评估区
        self.create_quality_metrics(control_frame_inside)
        
        # 进度条
        self.progress_bar = ttk.Progressbar(control_frame_inside, variable=self.progress, 
                                          maximum=100, length=250, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=5, pady=10)
        
        # 调整控制面板内部框架大小
        control_frame_inside.update_idletasks()
        control_canvas.config(scrollregion=control_canvas.bbox("all"))
        
        # 滚动功能
        def _on_mousewheel(event):
            control_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        # 只在鼠标悬停在控制面板上时响应滚轮
        def _on_enter(event):
            control_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _on_leave(event):
            control_canvas.unbind_all("<MouseWheel>")
        
        control_canvas.bind("<Enter>", _on_enter)
        control_canvas.bind("<Leave>", _on_leave)
        
        return control_outer_frame
    
    def create_file_operations(self, parent):
        """创建文件操作区域"""
        file_frame = ttk.LabelFrame(parent, text="文件操作")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 加载图像按钮
        load_btn = ttk.Button(file_frame, text="加载图像 (Ctrl+O)", 
                             command=self.load_image, style='Processing.TButton')
        load_btn.pack(fill=tk.X, padx=5, pady=2)
        
        # 保存按钮
        save_btn = ttk.Button(file_frame, text="保存结果 (Ctrl+S)", 
                             command=self.save_image, style='Processing.TButton')
        save_btn.pack(fill=tk.X, padx=5, pady=2)
        
        # 批量处理按钮
        batch_btn = ttk.Button(file_frame, text="批量处理", 
                              command=self.batch_process, style='Processing.TButton')
        batch_btn.pack(fill=tk.X, padx=5, pady=2)
    
    def create_noise_controls(self, parent):
        """创建噪声控制区域"""
        # 噪声类型选择
        noise_frame = ttk.LabelFrame(parent, text="噪声类型")
        noise_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.noise_type = tk.StringVar(value="gaussian")
        noise_types = [
            ("高斯噪声", "gaussian"),
            ("椒盐噪声", "salt_pepper"),
            ("泊松噪声", "poisson"),
            ("混合噪声", "mixed"),
            ("动态模糊", "motion_blur")
        ]
        
        for text, value in noise_types:
            ttk.Radiobutton(noise_frame, text=text, variable=self.noise_type, 
                           value=value, command=self.update_noise_params).pack(anchor=tk.W)
        
        # 噪声参数
        self.param_frame = ttk.LabelFrame(parent, text="噪声参数")
        self.param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 初始化所有参数控件
        self.create_all_noise_params()
        
        # 添加噪声按钮
        add_noise_btn = ttk.Button(parent, text="添加噪声", command=self.add_noise_async,
                                  style='Processing.TButton')
        add_noise_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 显示初始参数
        self.update_noise_params()
    
    def create_all_noise_params(self):
        """创建所有噪声参数控件"""
        # 高斯噪声参数
        self.gaussian_frame = ttk.Frame(self.param_frame)
        ttk.Label(self.gaussian_frame, text="标准差 (σ)").pack(anchor=tk.W)
        self.sigma_scale = Scale(self.gaussian_frame, from_=0, to=100, orient=HORIZONTAL, resolution=1)
        self.sigma_scale.set(25)
        self.sigma_scale.pack(fill=tk.X)
        self.sigma_label = ttk.Label(self.gaussian_frame, text="当前值: 25")
        self.sigma_label.pack(anchor=tk.W)
        self.sigma_scale.config(command=lambda v: self.sigma_label.config(text=f"当前值: {int(float(v))}"))
        
        # 椒盐噪声参数
        self.sp_frame = ttk.Frame(self.param_frame)
        ttk.Label(self.sp_frame, text="噪声密度").pack(anchor=tk.W)
        self.sp_scale = Scale(self.sp_frame, from_=0, to=0.5, orient=HORIZONTAL, resolution=0.01)
        self.sp_scale.set(0.05)
        self.sp_scale.pack(fill=tk.X)
        self.sp_label = ttk.Label(self.sp_frame, text="当前值: 0.05")
        self.sp_label.pack(anchor=tk.W)
        self.sp_scale.config(command=lambda v: self.sp_label.config(text=f"当前值: {float(v):.2f}"))
        
        # 泊松噪声参数
        self.poisson_frame = ttk.Frame(self.param_frame)
        ttk.Label(self.poisson_frame, text="噪声强度 (λ)").pack(anchor=tk.W)
        self.poisson_scale = Scale(self.poisson_frame, from_=1, to=50, orient=HORIZONTAL, resolution=1)
        self.poisson_scale.set(10)
        self.poisson_scale.pack(fill=tk.X)
        self.poisson_label = ttk.Label(self.poisson_frame, text="当前值: 10")
        self.poisson_label.pack(anchor=tk.W)
        self.poisson_scale.config(command=lambda v: self.poisson_label.config(text=f"当前值: {int(float(v))}"))
        
        # 动态模糊参数
        self.blur_frame = ttk.Frame(self.param_frame)
        ttk.Label(self.blur_frame, text="模糊长度").pack(anchor=tk.W)
        self.blur_length_scale = Scale(self.blur_frame, from_=1, to=20, orient=HORIZONTAL, resolution=1)
        self.blur_length_scale.set(5)
        self.blur_length_scale.pack(fill=tk.X)
        ttk.Label(self.blur_frame, text="模糊角度").pack(anchor=tk.W)
        self.blur_angle_scale = Scale(self.blur_frame, from_=0, to=180, orient=HORIZONTAL, resolution=1)
        self.blur_angle_scale.set(45)
        self.blur_angle_scale.pack(fill=tk.X)
    
    def update_noise_params(self):
        """根据选择的噪声类型更新参数显示"""
        # 隐藏所有参数框架
        for frame in [self.gaussian_frame, self.sp_frame, self.poisson_frame, self.blur_frame]:
            frame.pack_forget()
        
        # 显示对应的参数框架
        noise_type = self.noise_type.get()
        if noise_type == "gaussian":
            self.gaussian_frame.pack(fill=tk.X, padx=5, pady=5)
        elif noise_type == "salt_pepper":
            self.sp_frame.pack(fill=tk.X, padx=5, pady=5)
        elif noise_type == "poisson":
            self.poisson_frame.pack(fill=tk.X, padx=5, pady=5)
        elif noise_type == "motion_blur":
            self.blur_frame.pack(fill=tk.X, padx=5, pady=5)
        elif noise_type == "mixed":
            self.gaussian_frame.pack(fill=tk.X, padx=5, pady=5)
            self.sp_frame.pack(fill=tk.X, padx=5, pady=5)
    
    def create_denoise_controls(self, parent):
        """创建去噪控制区域"""
        denoise_frame = ttk.LabelFrame(parent, text="传统去噪方法")
        denoise_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 去噪参数标签页
        notebook = ttk.Notebook(denoise_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 均值滤波参数
        mean_tab = ttk.Frame(notebook)
        notebook.add(mean_tab, text="均值滤波")
        ttk.Label(mean_tab, text="窗口大小").pack(anchor=tk.W, padx=5, pady=2)
        self.mean_size = tk.IntVar(value=5)
        for size in [3, 5, 7, 9]:
            ttk.Radiobutton(mean_tab, text=f"{size}x{size}", variable=self.mean_size, 
                           value=size).pack(anchor=tk.W, padx=10)
        
        # 中值滤波参数
        median_tab = ttk.Frame(notebook)
        notebook.add(median_tab, text="中值滤波")
        ttk.Label(median_tab, text="窗口大小").pack(anchor=tk.W, padx=5, pady=2)
        self.median_size = tk.IntVar(value=5)
        for size in [3, 5, 7, 9]:
            ttk.Radiobutton(median_tab, text=f"{size}x{size}", variable=self.median_size, 
                           value=size).pack(anchor=tk.W, padx=10)
        
        # 小波变换参数
        wavelet_tab = ttk.Frame(notebook)
        notebook.add(wavelet_tab, text="小波变换")
        
        ttk.Label(wavelet_tab, text="小波基").pack(anchor=tk.W, padx=5, pady=2)
        self.wavelet_base = tk.StringVar(value="db4")
        bases = ["db1", "db2", "db4", "db8", "haar", "sym2", "sym4", "coif1", "bior1.3"]
        wavelet_combo = ttk.Combobox(wavelet_tab, textvariable=self.wavelet_base, values=bases, width=15)
        wavelet_combo.pack(fill=tk.X, padx=10, pady=2)
        
        ttk.Label(wavelet_tab, text="分解层数").pack(anchor=tk.W, padx=5, pady=2)
        self.wavelet_level = tk.IntVar(value=3)
        for level in range(2, 6):
            ttk.Radiobutton(wavelet_tab, text=f"{level}层", variable=self.wavelet_level, 
                           value=level).pack(anchor=tk.W, padx=10)
        
        ttk.Label(wavelet_tab, text="阈值模式").pack(anchor=tk.W, padx=5, pady=2)
        self.wavelet_mode = tk.StringVar(value="soft")
        ttk.Radiobutton(wavelet_tab, text="软阈值", variable=self.wavelet_mode, 
                       value="soft").pack(anchor=tk.W, padx=10)
        ttk.Radiobutton(wavelet_tab, text="硬阈值", variable=self.wavelet_mode, 
                       value="hard").pack(anchor=tk.W, padx=10)
        
        # 执行去噪按钮
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # 使用不同的颜色和图标（通过Unicode字符）
        mean_btn = tk.Button(button_frame, text="▶ 均值滤波去噪", bg="#e1f5fe", 
                            font=("Arial", 10, "bold"), command=lambda: self.denoise_async("mean"))
        mean_btn.pack(fill=tk.X, pady=2)
        
        median_btn = tk.Button(button_frame, text="▶ 中值滤波去噪", bg="#e8f5e9", 
                              font=("Arial", 10, "bold"), command=lambda: self.denoise_async("median"))
        median_btn.pack(fill=tk.X, pady=2)
        
        wavelet_btn = tk.Button(button_frame, text="▶ 小波变换去噪", bg="#fff3e0", 
                               font=("Arial", 10, "bold"), command=lambda: self.denoise_async("wavelet"))
        wavelet_btn.pack(fill=tk.X, pady=2)
    
    def create_deep_learning_controls(self, parent):
        """创建深度学习控制区域"""
        dl_frame = ttk.LabelFrame(parent, text="深度学习去噪方法")
        dl_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 模型状态显示
        self.model_status_label = ttk.Label(dl_frame, text="模型状态: 检查中...")
        self.model_status_label.pack(padx=5, pady=5)
        
        # 深度学习去噪按钮
        msresnet_btn = tk.Button(dl_frame, text="🧠 MS-ResNet去噪", bg="#e3f2fd", 
                                font=("Arial", 10, "bold"), 
                                command=lambda: self.denoise_deep_learning_async("ms_resnet"))
        msresnet_btn.pack(fill=tk.X, padx=5, pady=2)
        
        dae_btn = tk.Button(dl_frame, text="🧠 DAE-SPP去噪", bg="#f3e5f5", 
                           font=("Arial", 10, "bold"), 
                           command=lambda: self.denoise_deep_learning_async("dae_spp"))
        dae_btn.pack(fill=tk.X, padx=5, pady=2)
        
        # 比较所有方法按钮
        compare_btn = tk.Button(dl_frame, text="⚡ 比较所有方法", bg="#fce4ec", 
                               font=("Arial", 10, "bold"), command=self.compare_all_methods)
        compare_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 训练按钮（可选）
        train_frame = ttk.Frame(dl_frame)
        train_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(train_frame, text="模型训练（需要数据集）").pack(anchor=tk.W)
        train_btn = ttk.Button(train_frame, text="训练模型", command=self.train_models)
        train_btn.pack(fill=tk.X, pady=2)
        
        # 更新模型状态
        self.update_model_status()
    
    def update_model_status(self):
        """更新模型状态显示"""
        if self.models_loaded:
            self.model_status_label.config(text=f"模型状态: 已加载 (设备: {self.device})")
        else:
            self.model_status_label.config(text="模型状态: 未加载")
    
    def create_quality_metrics(self, parent):
        """创建图像质量评估区域"""
        quality_frame = ttk.LabelFrame(parent, text="图像质量评估")
        quality_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 使用网格布局显示指标
        metrics_frame = ttk.Frame(quality_frame)
        metrics_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(metrics_frame, textvariable=self.psnr_value).grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Label(metrics_frame, textvariable=self.ssim_value).grid(row=1, column=0, sticky=tk.W, padx=5)
        ttk.Label(metrics_frame, textvariable=self.mse_value).grid(row=2, column=0, sticky=tk.W, padx=5)
        ttk.Label(metrics_frame, textvariable=self.sam_value).grid(row=3, column=0, sticky=tk.W, padx=5)
        
        # 添加帮助信息
        help_text = "PSNR: 越高越好 (>30dB 良好)\nSSIM: 越接近1越好\nMSE: 越低越好\nSAM: 越低越好 (弧度)"
        help_label = ttk.Label(metrics_frame, text=help_text, font=("Arial", 8), foreground="gray")
        help_label.grid(row=0, column=1, rowspan=4, padx=10, sticky=tk.W)
    
    def create_image_display(self, parent):
        """创建图像显示区域"""
        image_frame = ttk.LabelFrame(parent, text="图像显示")
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建工具栏
        toolbar_frame = ttk.Frame(image_frame)
        toolbar_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # 视图选择按钮
        ttk.Button(toolbar_frame, text="2x2视图", 
                  command=lambda: self.change_view_layout("2x2")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="1x3视图", 
                  command=lambda: self.change_view_layout("1x3")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="单图视图", 
                  command=lambda: self.change_view_layout("single")).pack(side=tk.LEFT, padx=2)
        
        # 创建显示区
        self.fig = plt.Figure(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=image_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 初始化子图
        self.current_layout = "2x2"
        self.setup_subplots()
        
        return image_frame
    
    def create_status_bar(self):
        """创建状态栏"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = ttk.Label(status_frame, text="就绪", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 图像信息标签
        self.image_info_label = ttk.Label(status_frame, text="", relief=tk.SUNKEN)
        self.image_info_label.pack(side=tk.RIGHT, padx=5)
    
    def change_view_layout(self, layout):
        """改变视图布局"""
        self.current_layout = layout
        self.setup_subplots()
        self.refresh_display()
    
    def setup_subplots(self):
        """根据当前布局设置子图"""
        self.fig.clear()
        
        if self.current_layout == "2x2":
            # 创建2x2的子图布局
            self.ax1 = self.fig.add_subplot(221)
            self.ax2 = self.fig.add_subplot(222)
            self.ax3 = self.fig.add_subplot(223)
            self.ax4 = self.fig.add_subplot(224)
            
            self.ax1.set_title("原始图像")
            self.ax2.set_title("含噪图像")
            self.ax3.set_title("去噪结果")
            self.ax4.set_title("图像对比")
            
            self.axes = [self.ax1, self.ax2, self.ax3, self.ax4]
            
        elif self.current_layout == "1x3":
            # 创建1x3的子图布局
            self.ax1 = self.fig.add_subplot(131)
            self.ax2 = self.fig.add_subplot(132)
            self.ax3 = self.fig.add_subplot(133)
            
            self.ax1.set_title("原始图像")
            self.ax2.set_title("含噪图像")
            self.ax3.set_title("去噪结果")
            
            self.axes = [self.ax1, self.ax2, self.ax3]
            
        elif self.current_layout == "single":
            # 单图视图
            self.ax1 = self.fig.add_subplot(111)
            self.ax1.set_title("图像显示")
            self.axes = [self.ax1]
        
        for ax in self.axes:
            ax.axis('off')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def refresh_display(self):
        """刷新显示内容"""
        # 清除所有子图
        for ax in self.axes:
            ax.clear()
            ax.axis('off')
        
        # 根据当前布局重新显示图像
        if self.current_layout in ["2x2", "1x3"]:
            if self.original_img is not None:
                img_rgb = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
                self.ax1.imshow(img_rgb)
                self.ax1.set_title("原始图像")
            
            if self.noisy_img is not None and len(self.axes) > 1:
                noisy_rgb = cv2.cvtColor(self.noisy_img, cv2.COLOR_BGR2RGB)
                self.ax2.imshow(noisy_rgb)
                self.ax2.set_title("含噪图像")
            
            if self.denoised_img is not None and len(self.axes) > 2:
                denoised_rgb = cv2.cvtColor(self.denoised_img, cv2.COLOR_BGR2RGB)
                self.ax3.imshow(denoised_rgb)
                self.ax3.set_title("去噪结果")
            
            if self.current_layout == "2x2" and len(self.axes) > 3:
                # 显示对比图
                if self.noisy_img is not None and self.denoised_img is not None:
                    noisy_rgb = cv2.cvtColor(self.noisy_img, cv2.COLOR_BGR2RGB)
                    denoised_rgb = cv2.cvtColor(self.denoised_img, cv2.COLOR_BGR2RGB)
                    compared = np.hstack((noisy_rgb, denoised_rgb))
                    self.ax4.imshow(compared)
                    self.ax4.set_title("图像对比（含噪 vs 去噪）")
        
        self.canvas.draw()
    
    def load_image(self):
        """加载图像"""
        file_path = filedialog.askopenfilename(
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if file_path:
            try:
                self.original_img = cv2.imread(file_path)
                if self.original_img is None:
                    messagebox.showerror("错误", "无法读取图像文件！")
                    return
                
                # 更新图像信息
                h, w, c = self.original_img.shape
                self.image_info_label.config(text=f"图像: {w}x{h}, {c}通道")
                
                # 转换BGR到RGB用于显示
                img_rgb = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
                self.display_image(self.ax1, img_rgb, "原始图像")
                
                # 清除其他图像
                self.noisy_img = None
                self.denoised_img = None
                
                # 刷新显示
                self.refresh_display()
                
                # 重置评估指标
                self.reset_metrics()
                
                # 更新状态
                self.update_status(f"已加载图像: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("错误", f"加载图像时出错: {str(e)}")
    
    def add_noise_async(self):
        """异步添加噪声"""
        if self.original_img is None:
            messagebox.showwarning("提示", "请先加载图像！")
            return
        
        self.update_status("正在添加噪声...")
        self.progress.set(0)
        
        # 在后台线程中执行
        self.executor.submit(self.add_noise_worker)
    
    def add_noise_worker(self):
        """噪声添加工作线程"""
        try:
            # 更新进度
            self.root.after(0, lambda: self.progress.set(20))
            
            # 创建噪声图像的副本
            self.noisy_img = self.original_img.copy()
            
            noise_type = self.noise_type.get()
            
            # 更新进度
            self.root.after(0, lambda: self.progress.set(50))
            
            if noise_type == "gaussian":
                self.add_gaussian_noise()
            elif noise_type == "salt_pepper":
                self.add_salt_pepper_noise()
            elif noise_type == "poisson":
                self.add_poisson_noise()
            elif noise_type == "mixed":
                self.add_mixed_noise()
            elif noise_type == "motion_blur":
                self.add_motion_blur()
            
            # 更新进度
            self.root.after(0, lambda: self.progress.set(80))
            
            # 在主线程中更新UI
            self.root.after(0, self.update_noise_display)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"添加噪声时出错: {str(e)}"))
        finally:
            self.root.after(0, lambda: self.progress.set(100))
            self.root.after(500, lambda: self.progress.set(0))
    
    def add_gaussian_noise(self):
        """添加高斯噪声"""
        sigma = self.sigma_scale.get()
        gauss = np.random.normal(0, sigma, self.noisy_img.shape).astype(np.float32)
        self.noisy_img = np.clip(self.noisy_img + gauss, 0, 255).astype(np.uint8)
    
    def add_salt_pepper_noise(self):
        """添加椒盐噪声"""
        density = self.sp_scale.get()
        h, w, c = self.noisy_img.shape
        
        # 为椒噪声和盐噪声分别生成随机掩码
        salt_mask = np.random.random((h, w)) < density / 2
        pepper_mask = np.random.random((h, w)) < density / 2
        
        for i in range(c):
            self.noisy_img[salt_mask, i] = 255
            self.noisy_img[pepper_mask, i] = 0
    
    def add_poisson_noise(self):
        """添加泊松噪声"""
        lam = self.poisson_scale.get()
        norm_img = self.noisy_img / 255.0
        noisy_norm = np.random.poisson(norm_img * lam) / lam
        self.noisy_img = np.clip(noisy_norm * 255, 0, 255).astype(np.uint8)
    
    def add_mixed_noise(self):
        """添加混合噪声"""
        # 先添加高斯噪声
        self.add_gaussian_noise()
        
        # 再添加椒盐噪声（密度减半）
        original_density = self.sp_scale.get()
        self.sp_scale.set(original_density / 2)
        self.add_salt_pepper_noise()
        self.sp_scale.set(original_density)  # 恢复原值
    
    def add_motion_blur(self):
        """添加动态模糊"""
        length = self.blur_length_scale.get()
        angle = self.blur_angle_scale.get()
        
        # 创建运动模糊核
        kernel = self.create_motion_blur_kernel(length, angle)
        
        # 应用卷积
        for i in range(3):
            self.noisy_img[:,:,i] = cv2.filter2D(self.noisy_img[:,:,i], -1, kernel)
    
    @lru_cache(maxsize=32)
    def create_motion_blur_kernel(self, length, angle):
        """创建运动模糊核（带缓存）"""
        rad = np.deg2rad(angle)
        dx = np.cos(rad)
        dy = np.sin(rad)
        kernel_size = 2 * length + 1
        kernel = np.zeros((kernel_size, kernel_size))
        
        cx, cy = length, length
        for i in range(kernel_size):
            x = int(cx + dx * (i - length))
            y = int(cy + dy * (i - length))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1
        
        return kernel / np.sum(kernel)
    
    def update_noise_display(self):
        """更新噪声图像显示"""
        noisy_rgb = cv2.cvtColor(self.noisy_img, cv2.COLOR_BGR2RGB)
        self.display_image(self.ax2, noisy_rgb, "含噪图像")
        
        # 刷新显示
        self.refresh_display()
        
        # 重置评估指标
        self.reset_metrics()
        
        self.update_status("噪声添加完成")
    
    def denoise_async(self, method):
        """异步去噪处理"""
        if self.noisy_img is None:
            messagebox.showwarning("提示", "请先添加噪声！")
            return
        
        self.update_status(f"正在进行{method}去噪...")
        self.progress.set(0)
        
        # 在后台线程中执行
        self.executor.submit(self.denoise_worker, method)
    
    def denoise_worker(self, method):
        """去噪工作线程"""
        try:
            # 更新进度
            self.root.after(0, lambda: self.progress.set(30))
            
            if method == "mean":
                kernel_size = self.mean_size.get()
                self.denoised_img = cv2.blur(self.noisy_img, (kernel_size, kernel_size))
                title = f"均值滤波 ({kernel_size}x{kernel_size})"
                
            elif method == "median":
                kernel_size = self.median_size.get()
                self.denoised_img = cv2.medianBlur(self.noisy_img, kernel_size)
                title = f"中值滤波 ({kernel_size}x{kernel_size})"
                
            elif method == "wavelet":
                wavelet = self.wavelet_base.get()
                level = self.wavelet_level.get()
                mode = self.wavelet_mode.get()
                
                self.root.after(0, lambda: self.progress.set(50))
                
                self.denoised_img = self.wavelet_denoise_color(
                    self.noisy_img, wavelet=wavelet, level=level, mode=mode
                )
                title = f"小波变换去噪 ({wavelet}, {level}层, {mode}阈值)"
            
            # 更新进度
            self.root.after(0, lambda: self.progress.set(80))
            
            # 在主线程中更新UI
            self.root.after(0, lambda: self.update_denoise_display(title))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"去噪处理时出错: {str(e)}"))
        finally:
            self.root.after(0, lambda: self.progress.set(100))
            self.root.after(500, lambda: self.progress.set(0))
    
    def denoise_deep_learning_async(self, model_name):
        """异步深度学习去噪处理"""
        if self.noisy_img is None:
            messagebox.showwarning("提示", "请先添加噪声！")
            return
        
        if not self.models_loaded:
            messagebox.showwarning("提示", "深度学习模型未加载！")
            return
        
        self.update_status(f"正在进行{model_name}去噪...")
        self.progress.set(0)
        
        # 在后台线程中执行
        self.executor.submit(self.denoise_deep_learning_worker, model_name)
    
    def denoise_deep_learning_worker(self, model_name):
        """深度学习去噪工作线程"""
        try:
            # 更新进度
            self.root.after(0, lambda: self.progress.set(20))
            
            # 预处理图像
            noisy_tensor = torch.from_numpy(
                self.noisy_img.transpose(2, 0, 1)
            ).float().unsqueeze(0) / 255.0
            noisy_tensor = noisy_tensor.to(self.device)
            
            # 更新进度
            self.root.after(0, lambda: self.progress.set(50))
            
            # 选择模型并进行推理
            with torch.no_grad():
                if model_name == "ms_resnet":
                    denoised_tensor = self.ms_resnet(noisy_tensor)
                    title = "MS-ResNet去噪"
                elif model_name == "dae_spp":
                    denoised_tensor = self.dae_spp(noisy_tensor)
                    title = "DAE-SPP去噪"
                else:
                    raise ValueError(f"未知的模型: {model_name}")
            
            # 更新进度
            self.root.after(0, lambda: self.progress.set(80))
            
            # 后处理
            denoised_numpy = denoised_tensor.squeeze(0).cpu().numpy()
            denoised_numpy = np.transpose(denoised_numpy, (1, 2, 0))
            self.denoised_img = np.clip(denoised_numpy * 255, 0, 255).astype(np.uint8)
            
            # 在主线程中更新UI
            self.root.after(0, lambda: self.update_denoise_display(title))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"深度学习去噪时出错: {str(e)}"))
        finally:
            self.root.after(0, lambda: self.progress.set(100))
            self.root.after(500, lambda: self.progress.set(0))
    
    def update_denoise_display(self, title):
        """更新去噪结果显示"""
        denoised_rgb = cv2.cvtColor(self.denoised_img, cv2.COLOR_BGR2RGB)
        self.display_image(self.ax3, denoised_rgb, title)
        
        # 刷新显示
        self.refresh_display()
        
        # 计算评估指标
        self.calculate_metrics()
        
        self.update_status("去噪处理完成")
    
    def compare_all_methods(self):
        """比较所有去噪方法"""
        if self.noisy_img is None:
            messagebox.showwarning("提示", "请先添加噪声！")
            return
        
        # 创建比较窗口
        compare_window = tk.Toplevel(self.root)
        compare_window.title("去噪方法比较")
        compare_window.geometry("1400x900")
        
        # 创建图形
        fig = plt.Figure(figsize=(14, 9))
        canvas = FigureCanvasTkAgg(fig, master=compare_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 执行所有去噪方法
        methods = [
            ("原始图像", self.original_img),
            ("含噪图像", self.noisy_img),
            (f"均值滤波({self.mean_size.get()}x{self.mean_size.get()})", 
             cv2.blur(self.noisy_img, (self.mean_size.get(), self.mean_size.get()))),
            (f"中值滤波({self.median_size.get()}x{self.median_size.get()})", 
             cv2.medianBlur(self.noisy_img, self.median_size.get())),
            (f"小波变换({self.wavelet_base.get()})", 
             self.wavelet_denoise_color(self.noisy_img, self.wavelet_base.get(), 
                                       self.wavelet_level.get(), self.wavelet_mode.get()))
        ]
        
        # 添加深度学习方法（如果已加载）
        if self.models_loaded:
            # MS-ResNet
            noisy_tensor = torch.from_numpy(
                self.noisy_img.transpose(2, 0, 1)
            ).float().unsqueeze(0) / 255.0
            noisy_tensor = noisy_tensor.to(self.device)
            
            with torch.no_grad():
                # MS-ResNet去噪
                ms_resnet_out = self.ms_resnet(noisy_tensor)
                ms_resnet_img = ms_resnet_out.squeeze(0).cpu().numpy()
                ms_resnet_img = np.transpose(ms_resnet_img, (1, 2, 0))
                ms_resnet_img = np.clip(ms_resnet_img * 255, 0, 255).astype(np.uint8)
                methods.append(("MS-ResNet", ms_resnet_img))
                
                # DAE-SPP去噪
                dae_spp_out = self.dae_spp(noisy_tensor)
                dae_spp_img = dae_spp_out.squeeze(0).cpu().numpy()
                dae_spp_img = np.transpose(dae_spp_img, (1, 2, 0))
                dae_spp_img = np.clip(dae_spp_img * 255, 0, 255).astype(np.uint8)
                methods.append(("DAE-SPP", dae_spp_img))
        
        # 创建子图
        rows = 3
        cols = 3
        for i, (title, img) in enumerate(methods):
            if i < rows * cols:
                ax = fig.add_subplot(rows, cols, i+1)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
                    
                    # 计算PSNR和SSIM（除了原始图像）
                    if i > 0 and self.original_img is not None:
                        psnr = self.calculate_psnr(self.original_img, img)
                        ssim_val = self.calculate_ssim(
                            cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        )
                        sam = self.calculate_sam(self.original_img, img)
                        ax.set_title(f"{title}\nPSNR: {psnr:.2f} dB, SSIM: {ssim_val:.3f}\nSAM: {sam:.4f} rad", fontsize=9)
                    else:
                        ax.set_title(title)
                
                ax.axis('off')
        
        fig.tight_layout()
        canvas.draw()
    
    def calculate_metrics(self):
        """计算图像质量评估指标"""
        if self.original_img is not None and self.denoised_img is not None:
            # 计算PSNR
            psnr = self.calculate_psnr(self.original_img, self.denoised_img)
            self.psnr_value.set(f"PSNR: {psnr:.2f} dB")
            
            # 计算SSIM
            original_gray = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
            denoised_gray = cv2.cvtColor(self.denoised_img, cv2.COLOR_BGR2GRAY)
            ssim_val = self.calculate_ssim(original_gray, denoised_gray)
            self.ssim_value.set(f"SSIM: {ssim_val:.4f}")
            
            # 计算MSE
            mse = np.mean((self.original_img.astype(np.float32) - 
                          self.denoised_img.astype(np.float32)) ** 2)
            self.mse_value.set(f"MSE: {mse:.2f}")
            
            # 计算SAM
            sam = self.calculate_sam(self.original_img, self.denoised_img)
            self.sam_value.set(f"SAM: {sam:.4f} rad")
    
    def calculate_psnr(self, original, denoised):
        """计算PSNR"""
        if original.shape != denoised.shape:
            denoised = cv2.resize(denoised, (original.shape[1], original.shape[0]))
        
        mse = np.mean((original.astype(np.float32) - denoised.astype(np.float32)) ** 2)
        if mse == 0:
            return 100
        
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    def calculate_ssim(self, img1, img2, window_size=11, k1=0.01, k2=0.03, L=255):
        """计算SSIM（结构相似性指数）"""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # 使用skimage的SSIM实现
        return ssim(img1, img2, data_range=L)
    
    def calculate_sam(self, img1, img2):
        """计算光谱角映射（SAM）"""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # 将图像展平为向量
        img1_flat = img1.reshape(-1, img1.shape[2])
        img2_flat = img2.reshape(-1, img2.shape[2])
        
        # 计算每个像素的光谱角
        dots = np.sum(img1_flat * img2_flat, axis=1)
        norms1 = np.linalg.norm(img1_flat, axis=1)
        norms2 = np.linalg.norm(img2_flat, axis=1)
        
        # 避免除零
        norms = norms1 * norms2
        norms[norms == 0] = 1e-8
        
        # 计算角度
        cos_angles = np.clip(dots / norms, -1, 1)
        angles = np.arccos(cos_angles)
        
        # 返回平均光谱角
        return np.mean(angles)