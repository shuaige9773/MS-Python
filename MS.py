import os
import cv2
import numpy as np
import pywt
import tkinter as tk
from tkinter import filedialog, ttk, Scale, HORIZONTAL, Scrollbar
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import platform

class ImageDenoisingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("多光谱图像噪声添加与去噪处理")
        self.root.geometry("1280x800")  #窗口大小
        
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
        
        self.setup_ui()
    
    def setup_chinese_font(self):
        # 根据操作系统设置合适的中文字体
        system = platform.system()
        if system == 'Windows':
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
        elif system == 'Darwin':  # macOS
            plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC']
        else:  # Linux
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC']
        
        plt.rcParams['axes.unicode_minus'] = False  
    
    def setup_ui(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板 - 带滚动条
        control_outer_frame = ttk.Frame(main_frame)
        control_outer_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
        
        # 添加垂直滚动条
        v_scrollbar = Scrollbar(control_outer_frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 创建画布用于滚动
        control_canvas = tk.Canvas(control_outer_frame, yscrollcommand=v_scrollbar.set)
        control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 配置滚动条
        v_scrollbar.config(command=control_canvas.yview)
        
        # 创建内部框架用于放置控件
        control_frame_inside = ttk.Frame(control_canvas)
        control_canvas_window = control_canvas.create_window((0, 0), window=control_frame_inside, anchor='nw')
        
        # 设置控制面板的标题
        control_label = ttk.Label(control_frame_inside, text="控制面板", font=("Arial", 12, "bold"))
        control_label.pack(padx=5, pady=5)
        
        # 加载图像按钮
        ttk.Button(control_frame_inside, text="加载图像", command=self.load_image).pack(fill=tk.X, padx=5, pady=5)
        
        # 噪声类型选择
        noise_frame = ttk.LabelFrame(control_frame_inside, text="噪声类型")
        noise_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.noise_type = tk.StringVar(value="gaussian")
        ttk.Radiobutton(noise_frame, text="高斯噪声", variable=self.noise_type, value="gaussian").pack(anchor=tk.W)
        ttk.Radiobutton(noise_frame, text="椒盐噪声", variable=self.noise_type, value="salt_pepper").pack(anchor=tk.W)
        ttk.Radiobutton(noise_frame, text="泊松噪声", variable=self.noise_type, value="poisson").pack(anchor=tk.W)
        ttk.Radiobutton(noise_frame, text="混合噪声", variable=self.noise_type, value="mixed").pack(anchor=tk.W)
        ttk.Radiobutton(noise_frame, text="动态模糊", variable=self.noise_type, value="motion_blur").pack(anchor=tk.W)
        
        # 噪声参数
        param_frame = ttk.LabelFrame(control_frame_inside, text="噪声参数")
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 高斯噪声标准差
        ttk.Label(param_frame, text="高斯噪声标准差 (σ)").pack(anchor=tk.W)
        self.sigma_scale = Scale(param_frame, from_=0, to=100, orient=HORIZONTAL, resolution=1)
        self.sigma_scale.set(25)
        self.sigma_scale.pack(fill=tk.X)
        
        # 椒盐噪声密度
        ttk.Label(param_frame, text="椒盐噪声密度").pack(anchor=tk.W)
        self.sp_scale = Scale(param_frame, from_=0, to=1, orient=HORIZONTAL, resolution=0.01)
        self.sp_scale.set(0.05)
        self.sp_scale.pack(fill=tk.X)
        
        # 泊松噪声强度
        ttk.Label(param_frame, text="泊松噪声强度 (λ)").pack(anchor=tk.W)
        self.poisson_scale = Scale(param_frame, from_=1, to=50, orient=HORIZONTAL, resolution=1)
        self.poisson_scale.set(10)
        self.poisson_scale.pack(fill=tk.X)
        
        # 动态模糊长度
        ttk.Label(param_frame, text="动态模糊长度").pack(anchor=tk.W)
        self.blur_length_scale = Scale(param_frame, from_=1, to=20, orient=HORIZONTAL, resolution=1)
        self.blur_length_scale.set(5)
        self.blur_length_scale.pack(fill=tk.X)
        
        # 动态模糊角度
        ttk.Label(param_frame, text="动态模糊角度").pack(anchor=tk.W)
        self.blur_angle_scale = Scale(param_frame, from_=0, to=180, orient=HORIZONTAL, resolution=1)
        self.blur_angle_scale.set(45)
        self.blur_angle_scale.pack(fill=tk.X)
        
        # 添加噪声按钮
        ttk.Button(control_frame_inside, text="添加噪声", command=self.add_noise).pack(fill=tk.X, padx=5, pady=5)
        
        # 去噪方法选择
        denoise_frame = ttk.LabelFrame(control_frame_inside, text="去噪方法")
        denoise_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 去噪参数
        # 均值滤波窗口大小
        ttk.Label(denoise_frame, text="均值滤波窗口大小").pack(anchor=tk.W)
        self.mean_size = tk.IntVar(value=5)
        ttk.Radiobutton(denoise_frame, text="3x3", variable=self.mean_size, value=3).pack(anchor=tk.W)
        ttk.Radiobutton(denoise_frame, text="5x5", variable=self.mean_size, value=5).pack(anchor=tk.W)
        ttk.Radiobutton(denoise_frame, text="7x7", variable=self.mean_size, value=7).pack(anchor=tk.W)
        
        # 中值滤波窗口大小
        ttk.Label(denoise_frame, text="中值滤波窗口大小").pack(anchor=tk.W)
        self.median_size = tk.IntVar(value=5)
        ttk.Radiobutton(denoise_frame, text="3x3", variable=self.median_size, value=3).pack(anchor=tk.W)
        ttk.Radiobutton(denoise_frame, text="5x5", variable=self.median_size, value=5).pack(anchor=tk.W)
        ttk.Radiobutton(denoise_frame, text="7x7", variable=self.median_size, value=7).pack(anchor=tk.W)
        
        # 小波变换参数
        wavelet_frame = ttk.LabelFrame(denoise_frame, text="小波变换参数")
        wavelet_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(wavelet_frame, text="小波基").pack(anchor=tk.W)
        self.wavelet_base = tk.StringVar(value="db4")
        bases = ["db1", "db2", "db4", "haar", "sym2", "coif1"]
        ttk.Combobox(wavelet_frame, textvariable=self.wavelet_base, values=bases).pack(fill=tk.X)
        
        ttk.Label(wavelet_frame, text="分解层数").pack(anchor=tk.W)
        self.wavelet_level = tk.IntVar(value=3)
        ttk.Radiobutton(wavelet_frame, text="2层", variable=self.wavelet_level, value=2).pack(anchor=tk.W)
        ttk.Radiobutton(wavelet_frame, text="3层", variable=self.wavelet_level, value=3).pack(anchor=tk.W)
        ttk.Radiobutton(wavelet_frame, text="4层", variable=self.wavelet_level, value=4).pack(anchor=tk.W)
        
        ttk.Label(wavelet_frame, text="阈值模式").pack(anchor=tk.W)
        self.wavelet_mode = tk.StringVar(value="soft")
        ttk.Radiobutton(wavelet_frame, text="软阈值", variable=self.wavelet_mode, value="soft").pack(anchor=tk.W)
        ttk.Radiobutton(wavelet_frame, text="硬阈值", variable=self.wavelet_mode, value="hard").pack(anchor=tk.W)
        
        # 执行去噪按钮 - 使用醒目的颜色
        button_frame = ttk.Frame(control_frame_inside)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        

        mean_btn = tk.Button(button_frame, text="均值滤波去噪", bg="#e1f5fe", font=("Arial", 10, "bold"),
                             command=lambda: self.denoise("mean"))
        mean_btn.pack(fill=tk.X, pady=2)
        
        median_btn = tk.Button(button_frame, text="中值滤波去噪", bg="#e8f5e9", font=("Arial", 10, "bold"),
                               command=lambda: self.denoise("median"))
        median_btn.pack(fill=tk.X, pady=2)
        
        wavelet_btn = tk.Button(button_frame, text="小波变换去噪", bg="#fff3e0", font=("Arial", 10, "bold"),
                                command=lambda: self.denoise("wavelet"))
        wavelet_btn.pack(fill=tk.X, pady=2)
        
        # 图像质量评估显示区域
        quality_frame = ttk.LabelFrame(control_frame_inside, text="图像质量评估")
        quality_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(quality_frame, textvariable=self.psnr_value).pack(anchor=tk.W)
        ttk.Label(quality_frame, textvariable=self.ssim_value).pack(anchor=tk.W)
        
        # 保存按钮
        ttk.Button(control_frame_inside, text="保存结果", command=self.save_image).pack(fill=tk.X, padx=5, pady=5)
        
        # 右侧图像显示区域
        image_frame = ttk.LabelFrame(main_frame, text="图像显示")
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建显示区
        self.fig = plt.Figure(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=image_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 初始化子图
        self.setup_subplots()
        
        # 调整控制面板内部框架大小
        control_frame_inside.update_idletasks()
        control_canvas.config(scrollregion=control_canvas.bbox("all"))
        
        # 设置Canvas最小宽度
        control_canvas.config(width=280)
        
        # 滚动条
        def _on_mousewheel(event):
            control_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        control_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # 绑定事件以调整滚动范围
        def _configure_canvas(event):
            control_canvas.config(scrollregion=control_canvas.bbox("all"))
        control_frame_inside.bind("<Configure>", _configure_canvas)
    
    def setup_subplots(self):
        self.fig.clear()
        # 创建2x2的子图布局
        self.ax1 = self.fig.add_subplot(221)
        self.ax2 = self.fig.add_subplot(222)
        self.ax3 = self.fig.add_subplot(223)
        self.ax4 = self.fig.add_subplot(224)
        
        self.ax1.set_title("原始图像")
        self.ax2.set_title("含噪图像")
        self.ax3.set_title("去噪结果")
        self.ax4.set_title("图像对比（原始 vs 去噪）")
        
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.axis('off')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if file_path:
            self.original_img = cv2.imread(file_path)
            if self.original_img is None:
                print("错误：图像读取失败！")
                return
            
            # 转换BGR到RGB用于显示
            img_rgb = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
            self.display_image(self.ax1, img_rgb, "原始图像")
            
            # 清除其他子图
            self.ax2.clear()
            self.ax2.set_title("含噪图像")
            self.ax3.clear()
            self.ax3.set_title("去噪结果")
            self.ax4.clear()
            self.ax4.set_title("图像对比（原始 vs 去噪）")
            
            for ax in [self.ax2, self.ax3, self.ax4]:
                ax.axis('off')
            
            self.canvas.draw()
            
            # 重置评估指标
            self.psnr_value.set("PSNR: N/A")
            self.ssim_value.set("SSIM: N/A")
            
            # 重置其他图像变量
            self.noisy_img = None
            self.denoised_img = None
    
    def add_noise(self):
        if self.original_img is None:
            print("请先加载图像！")
            return
        
        # 创建噪声图像的副本
        self.noisy_img = self.original_img.copy()
        
        noise_type = self.noise_type.get()
        
        if noise_type == "gaussian":
            # 添加高斯噪声
            sigma = self.sigma_scale.get()
            gauss = np.random.normal(0, sigma, self.noisy_img.shape).astype(np.float32)
            self.noisy_img = np.clip(self.noisy_img + gauss, 0, 255).astype(np.uint8)
            
        elif noise_type == "salt_pepper":
            # 添加椒盐噪声
            density = self.sp_scale.get()
            h, w, c = self.noisy_img.shape
            
            # 椒盐噪声
            mask = np.random.random((h, w)) < density
            for i in range(c):
                # 盐噪声 (白点)
                salt = np.copy(mask)
                self.noisy_img[salt, i] = 255
                
                # 椒噪声 (黑点)
                pepper = np.copy(mask)
                self.noisy_img[pepper, i] = 0
            
        elif noise_type == "poisson":
            # 添加泊松噪声
            lam = self.poisson_scale.get()
            # 将图像归一化到[0,1]
            norm_img = self.noisy_img / 255.0
            # 根据图像强度生成泊松分布噪声
            noisy_norm = np.random.poisson(norm_img * lam) / lam
            # 转回[0,255]范围
            self.noisy_img = np.clip(noisy_norm * 255, 0, 255).astype(np.uint8)
            
        elif noise_type == "mixed":
            # 混合噪声 (高斯 + 椒盐)
            sigma = self.sigma_scale.get()
            density = self.sp_scale.get() / 2  # 降低椒盐密度以平衡总体噪声效果
    
            # 1. 先添加高斯噪声
            gauss = np.random.normal(0, sigma, self.noisy_img.shape).astype(np.float32)
            self.noisy_img = np.clip(self.noisy_img + gauss, 0, 255).astype(np.uint8)
    
            # 2. 再添加椒盐噪声 - 修复黑白点分布
            h, w, c = self.noisy_img.shape
    
            # 为椒噪声(黑点)和盐噪声(白点)分别生成随机掩码
            salt_mask = np.random.random((h, w)) < density / 2  # 盐噪声(白点)
            pepper_mask = np.random.random((h, w)) < density / 2  # 椒噪声(黑点)
    
            for i in range(c):
                # 添加盐噪声(白点)
                self.noisy_img[salt_mask, i] = 255
        
                # 添加椒噪声(黑点)
                self.noisy_img[pepper_mask, i] = 0
                
        elif noise_type == "motion_blur":
            # 动态模糊 (运动模糊)
            length = self.blur_length_scale.get()
            angle = self.blur_angle_scale.get()
            
            # 创建运动模糊核
            rad = np.deg2rad(angle)
            dx = np.cos(rad)
            dy = np.sin(rad)
            kernel_size = 2 * length + 1
            kernel = np.zeros((kernel_size, kernel_size))
            
            # 设置模糊核的中心线
            cx, cy = length, length
            for i in range(kernel_size):
                x = int(cx + dx * (i - length))
                y = int(cy + dy * (i - length))
                if 0 <= x < kernel_size and 0 <= y < kernel_size:
                    kernel[y, x] = 1
            
            # 归一化核
            kernel = kernel / np.sum(kernel)
            
            # 应用卷积
            for i in range(3):  # 对每个通道应用
                self.noisy_img[:,:,i] = cv2.filter2D(self.noisy_img[:,:,i], -1, kernel)
        
        # 显示含噪图像
        noisy_rgb = cv2.cvtColor(self.noisy_img, cv2.COLOR_BGR2RGB)
        self.display_image(self.ax2, noisy_rgb, "含噪图像")
        
        # 清除去噪结果
        self.ax3.clear()
        self.ax3.set_title("去噪结果")
        self.ax3.axis('off')
        
        # 更新对比图
        self.ax4.clear()
        self.ax4.set_title("图像对比（原始 vs 含噪）")
        self.ax4.axis('off')
        
        # 拼接原始图像和含噪图像进行对比
        if self.original_img is not None and self.noisy_img is not None:
            original_rgb = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
            compared = np.hstack((original_rgb, noisy_rgb))
            self.ax4.imshow(compared)
        
        self.canvas.draw()
        
        # 重置评估指标
        self.psnr_value.set("PSNR: N/A")
        self.ssim_value.set("SSIM: N/A")
    
    def denoise(self, method):
        if self.noisy_img is None:
            print("请先添加噪声！")
            return
        
        if method == "mean":
            # 均值滤波
            kernel_size = self.mean_size.get()
            self.denoised_img = cv2.blur(self.noisy_img, (kernel_size, kernel_size))
            title = f"均值滤波 ({kernel_size}x{kernel_size})"
            
        elif method == "median":
            # 中值滤波
            kernel_size = self.median_size.get()
            self.denoised_img = cv2.medianBlur(self.noisy_img, kernel_size)
            title = f"中值滤波 ({kernel_size}x{kernel_size})"
            
        elif method == "wavelet":
            # 小波变换去噪
            wavelet = self.wavelet_base.get()
            level = self.wavelet_level.get()
            mode = self.wavelet_mode.get()
            
            self.denoised_img = self.wavelet_denoise_color(
                self.noisy_img, wavelet=wavelet, level=level, mode=mode
            )
            title = f"小波变换去噪 ({wavelet}, {level}层, {mode}阈值)"
        
        # 显示去噪结果
        denoised_rgb = cv2.cvtColor(self.denoised_img, cv2.COLOR_BGR2RGB)
        self.display_image(self.ax3, denoised_rgb, title)
        
        # 更新对比图
        self.ax4.clear()
        self.ax4.set_title("图像对比（含噪 vs 去噪）")
        self.ax4.axis('off')
        
        # 拼接含噪图像和去噪图像进行对比
        if self.noisy_img is not None and self.denoised_img is not None:
            noisy_rgb = cv2.cvtColor(self.noisy_img, cv2.COLOR_BGR2RGB)
            compared = np.hstack((noisy_rgb, denoised_rgb))
            self.ax4.imshow(compared)
        
        self.canvas.draw()
        
        # 计算评估指标
        self.calculate_metrics()
    
    def calculate_metrics(self):
        if self.original_img is not None and self.denoised_img is not None:
            # 计算PSNR
            psnr = self.calculate_psnr(self.original_img, self.denoised_img)
            self.psnr_value.set(f"PSNR: {psnr:.2f} dB")
            
            # 计算SSIM (需要转换为灰度图像)
            original_gray = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
            denoised_gray = cv2.cvtColor(self.denoised_img, cv2.COLOR_BGR2GRAY)
            ssim = self.calculate_ssim(original_gray, denoised_gray)
            self.ssim_value.set(f"SSIM: {ssim:.4f}")
    
    def calculate_psnr(self, original, denoised):
        # 确保图像大小一致
        if original.shape != denoised.shape:
            denoised = cv2.resize(denoised, (original.shape[1], original.shape[0]))
        
        # 计算MSE
        mse = np.mean((original.astype(np.float32) - denoised.astype(np.float32)) ** 2)
        if mse == 0:
            return 100  # 避免除以零
        
        # 计算PSNR
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    def calculate_ssim(self, img1, img2, window_size=11, k1=0.01, k2=0.03, L=255):
        # 简化版SSIM计算
        # 首先确保图像大小一致
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
        # 创建高斯窗口
        gauss_kernel = cv2.getGaussianKernel(window_size, 1.5)
        gauss_window = gauss_kernel * gauss_kernel.T
        
        # 计算均值
        mu1 = cv2.filter2D(img1.astype(np.float32), -1, gauss_window)[window_size//2:-window_size//2, window_size//2:-window_size//2]
        mu2 = cv2.filter2D(img2.astype(np.float32), -1, gauss_window)[window_size//2:-window_size//2, window_size//2:-window_size//2]
        
        # 计算方差和协方差
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.filter2D(img1.astype(np.float32) * img1.astype(np.float32), -1, gauss_window)[window_size//2:-window_size//2, window_size//2:-window_size//2] - mu1_sq
        sigma2_sq = cv2.filter2D(img2.astype(np.float32) * img2.astype(np.float32), -1, gauss_window)[window_size//2:-window_size//2, window_size//2:-window_size//2] - mu2_sq
        sigma12 = cv2.filter2D(img1.astype(np.float32) * img2.astype(np.float32), -1, gauss_window)[window_size//2:-window_size//2, window_size//2:-window_size//2] - mu1_mu2
        
        # 常数项
        C1 = (k1 * L) ** 2
        C2 = (k2 * L) ** 2
        
        # 计算SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # 返回SSIM平均值
        return np.mean(ssim_map)
    
    def wavelet_denoise_color(self, img, wavelet='db4', level=3, mode='soft'):
        # 创建输出图像
        output = np.zeros_like(img)
        
        # 对每个通道单独处理
        for i in range(3):  # BGR通道
            # 提取单个通道
            channel = img[:,:,i]
            
            # 小波分解
            coeffs = pywt.wavedec2(channel, wavelet, level=level)
            
            # 计算通用阈值（VisuShrink方法）
            sigma = np.median(np.abs(coeffs[-1] - np.median(coeffs[-1]))) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(channel.size))
            
            # 阈值处理（除近似系数外）
            coeffs_thresh = [coeffs[0]]  # 保留近似系数
            for j in range(1, len(coeffs)):
                coeffs_thresh.append(
                    tuple(pywt.threshold(c, threshold, mode=mode) for c in coeffs[j])
                )
            
            # 小波重构
            denoised = pywt.waverec2(coeffs_thresh, wavelet)
            # 确保尺寸一致
            if denoised.shape != channel.shape:
                denoised = denoised[:channel.shape[0], :channel.shape[1]]
            output[:,:,i] = np.clip(denoised, 0, 255).astype(np.uint8)
        
        return output
    
    def display_image(self, ax, img, title):
        ax.clear()
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
        self.canvas.draw()
    
    def save_image(self):
        if self.denoised_img is None:
            print("请先执行去噪操作！")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG文件", "*.png"), ("JPEG文件", "*.jpg"), ("BMP文件", "*.bmp")]
        )
        
        if file_path:
            cv2.imwrite(file_path, self.denoised_img)
            print(f"图像已保存至: {file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageDenoisingApp(root)
    root.mainloop()
