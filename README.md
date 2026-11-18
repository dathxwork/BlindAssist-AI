# BlindAssist AI - Hệ Thống Trợ Lý Thị Giác Máy Tính & Cảnh Báo Chướng Ngại Vật Hỗ Trợ Người Khiếm Thị

> **Đồ Án Nhóm Nghiên Cứu & Phát Triển Ứng Dụng AI Hỗ Trợ Cộng Đồng**  
> Ứng dụng mô hình học sâu **YOLO (Ultralytics)**, thị giác máy tính **OpenCV**, mô hình quang học lỗ kim **Pinhole Camera Model** và hệ thống cảnh báo đa phương tiện (Giọng nói tiếng Việt + Âm thanh Radar tiệm cận).

---

## 🚀 HƯỚNG DẪN CHẠY DỰ ÁN (CÁCH KHỞI CHẠY)

Hệ thống được đóng gói hoàn chỉnh với **2 nền tảng sử dụng**:

### 🌐 CÁCH 1: Khởi Chạy Giao Diện Web AI Chính (Khuyên dùng - Đầy đủ tính năng nhất)

Giao diện Web AI là sản phẩm hoàn chỉnh nhất của nhóm, hỗ trợ đầy đủ nhận diện thời gian thực, điều khiển âm lượng laptop, phân tích video và chuẩn trợ năng cho người khiếm thị.

#### Cách 1.1: Nhấp đúp chuột vào file khởi chạy tự động (Nhanh nhất)
* Nhấp đúp vào file **`run_web.bat`** (nằm ngay thư mục gốc dự án).
* Trình duyệt sẽ tự động mở trang web tại địa chỉ: **`http://localhost:8000`**

#### Cách 1.2: Chạy bằng câu lệnh trong Terminal / Command Prompt
Mở Terminal tại thư mục dự án và thực thi:
```bash
python -m uvicorn web_app:app --host 0.0.0.0 --port 8000
```
Sau đó mở trình duyệt (Google Chrome, Edge, Safari hoặc điện thoại cùng mạng Wi-Fi) truy cập vào:
👉 **[http://localhost:8000](http://localhost:8000)** (hoặc `http://127.0.0.1:8000`)

---

### 🖥️ CÁCH 2: Khởi Chạy Ứng Dụng Desktop (Xem trước trực tiếp bằng OpenCV)

Dành cho môi trường kiểm thử trực tiếp trên máy trạm hoặc nhúng vào kính thông minh / Raspberry Pi.

#### Cách 2.1: Nhấp đúp chuột
* Nhấp đúp vào file **`run_desktop.bat`** trong thư mục dự án.

#### Cách 2.2: Chạy bằng dòng lệnh CLI phong phú
```bash
# 1. Chạy với video kiểm thử đường phố mặc định:
python main.py

# 2. Chạy với Webcam trực tiếp thời gian thực:
python main.py --source 0

# 3. Tùy chỉnh mức âm lượng loa (ví dụ 80%):
python main.py --volume 0.8

# 4. Tùy chỉnh ngưỡng cự ly cảnh báo sớm (ví dụ 15 mét):
python main.py --alert-dist 15.0

# 5. Đổi sang âm thanh Tiếng Anh:
python main.py --lang en

# 6. Lưu video kết quả phân tích ra file:
python main.py --source test_video.mp4 --save ket_qua_xuly.mp4
```

---

## 💻 CÔNG NGHỆ, NGÔN NGỮ & THƯ VIỆN SỬ DỤNG

* **Ngôn ngữ lập trình chính:**
  * **Python 3.9+**: Xử lý toàn bộ logic hệ thống, thuật toán thị giác máy tính, tính toán hình học quang học và kiến trúc đa luồng (`Multi-threading`).
* **Trí tuệ nhân tạo (AI) & Deep Learning:**
  * **Ultralytics YOLO (YOLOv8 / YOLO11 / YOLO26)**: Mô hình học sâu phát hiện vật cản thời gian thực với độ chính xác cao và tốc độ khung hình mượt mà (> 30 FPS).
  * **PyTorch (`torch`, `torchvision`)**: Framework nạp và tối ưu hóa trọng số mô hình (`best.pt`, `gpModel.pt`).
* **Thị giác máy tính & Xử lý ảnh:**
  * **OpenCV (`opencv-python`)**: Thu nhận luồng video/webcam, trích xuất bounding box, vẽ HUD, vạch phân chia làn đường và làm mờ khuôn mặt bảo mật danh tính (Gaussian Blur).
  * **NumPy**: Xử lý đại số tuyến tính, ma trận điểm ảnh và giải thuật khoảng cách hình học quang học lỗ kim (Pinhole Camera Model).
* **Hệ thống Âm thanh & Tổng hợp giọng nói (Text-To-Speech):**
  * **pyttsx3 & Windows SAPI5**: Tổng hợp giọng nói cảnh báo ngoại tuyến (100% Offline TTS), chạy trên luồng nền (`daemon=True`) không gây giật lag hình ảnh.
  * **Web Speech API & gTTS**: Phát âm thanh cảnh báo bằng giọng nói tiếng Việt tự nhiên, truyền cảm trên giao diện web.
  * **Web Audio API**: Bộ tổng hợp xung âm thanh Radar / Sonar tiệm cận (phát tiếng bíp dồn dập khi vật thể tiến sát dưới 1.5m).
* **Giao diện Web & Mạng:**
  * **FastAPI & Uvicorn**: Backend REST API và WebSocket bất đồng bộ siêu tốc, độ trễ truyền hình ảnh dưới 100ms.
  * **HTML5, CSS3, JavaScript (ES6+)**: Giao diện website hiện đại, tương thích máy tính và điện thoại, tích hợp thanh trượt âm lượng laptop và chuẩn trợ năng WCAG 2.1 AAA.
* **Quản lý phiên bản & Môi trường:**
  * **Git**: Quản lý lịch sử mã nguồn dự án.
  * **pip / venv**: Đóng gói các thư viện phụ thuộc (`requirements.txt`).

---

## 🧠 NGUYÊN LÝ HOẠT ĐỘNG & KIẾN TRÚC HỆ THỐNG

### 1. Sơ đồ luồng xử lý dữ liệu (Pipeline Architecture)
```text
[Camera/Webcam/Video]
          │
          ▼
 [Khung hình Frame] ─────► [Làm mờ mặt người đi bộ (Gaussian Blur)]
          │
          ▼
[Mô hình AI YOLOv8] ─────► Trích xuất Bounding Box [x1, y1, x2, y2], Class, Conf
          │
          ├─────────────────────────────────────────────┐
          ▼                                             ▼
[Ước lượng khoảng cách hình học]             [Định vị phương hướng 3 vùng]
 d = (W_frame * f * ratio) / W_box           Tâm vật thể X_center = (x1 + x2) / 2
          │                                             │
          └──────────────────────┬──────────────────────┘
                                 ▼
              [Phân loại mức độ rủi ro va chạm]
                (Rất gần / Gần / Trung bình / Xa)
                                 │
                                 ▼
           [Bộ điều phối cảnh báo âm thanh đa luồng]
                ├─► Giọng nói Tiếng Việt: "{Tên} ở {Hướng}, {Khoảng cách} mét"
                ├─► Âm thanh Radar/Sonar: Bíp bíp dồn dập tiệm cận (< 1.5m)
                └─► Bộ đệm Cooldown: Chống đọc lặp, chống spam âm thanh
                                 │
                                 ▼
           [Hiển thị kết quả ra Web & Ứng dụng Desktop]
```

### 2. Thuật toán ước lượng khoảng cách (Pinhole Distance Estimation)
Khoảng cách $d$ (mét) từ camera đến vật thể được tính theo công thức quang học lỗ kim:
$$d = \frac{W_{\text{frame}} \times \left(\frac{0.5}{\tan(\text{FOV} / 2)}\right) \times \text{width\_ratio}}{W_{\text{box}} + 10^{-6}}$$
Trong đó:
* $W_{\text{frame}}$: Chiều rộng khung hình camera (pixels).
* $\text{FOV}$: Góc mở ngang của ống kính camera (mặc định $\approx 70^\circ$).
* $W_{\text{box}}$: Chiều rộng bounding box nhận diện được ($x_2 - x_1$).
* $\text{width\_ratio}$: Hệ số tỷ lệ kích thước thực tế chuẩn hóa của từng loại đối tượng so với con người.

### 3. Thuật toán phân vùng không gian định hướng (3 Zones)
Dựa vào tọa độ tâm đối tượng $X_{\text{center}} = \frac{x_1 + x_2}{2}$:
* **BÊN TRÁI (LEFT):** $X_{\text{center}} < 35\% \times W_{\text{frame}}$
* **PHÍA TRƯỚC (AHEAD):** $35\% \times W_{\text{frame}} \le X_{\text{center}} \le 65\% \times W_{\text{frame}}$
* **BÊN PHẢI (RIGHT):** $X_{\text{center}} > 65\% \times W_{\text{frame}}$

### 4. Thang đo 4 cấp độ khoảng cách & đánh giá rủi ro
| Cấp độ | Cự ly ước lượng | Mức độ rủi ro | Hành vi cảnh báo của hệ thống |
| :---: | :---: | :---: | :--- |
| **Rất gần** | $d \le 1.5\text{ m}$ | 🔴 Nguy cơ va chạm cao | Phát cảnh báo khẩn cấp tức thì (Radar bíp dồn dập + khung đỏ). |
| **Gần** | $1.5\text{ m} < d \le 3.0\text{ m}$ | 🟡 Cảnh giác | Nhắc nhở người dùng chú ý giảm tốc độ hoặc chuẩn bị đổi hướng. |
| **Trung bình** | $3.0\text{ m} < d \le 5.0\text{ m}$ | 🔵 Lưu ý an toàn | Vật cản ở cự ly an toàn đang tiến lại gần. |
| **Xa** | $d > 5.0\text{ m}$ | 🟢 An toàn | Ngoài phạm vi nguy hiểm, không gây cản trở di chuyển. |

---

## 🎯 BỘ DỮ LIỆU & CÁC LỚP VẬT CẢN HỖ TRỢ

Hệ thống hỗ trợ 2 bộ mô hình học sâu linh hoạt:

### 1. Bộ 10 lớp vật cản đường phố Việt Nam (Mô hình `best.pt`):
Được nhóm thu thập và huấn luyện chuyên biệt trên điều kiện giao thông và lối đi vỉa hè tại Việt Nam:

| STT | Tên lớp (Model) | Tên tiếng Việt | Chiều cao chuẩn | Phân loại | Ý nghĩa cảnh báo cho người khiếm thị |
| :---: | :--- | :--- | :---: | :---: | :--- |
| 1 | **Nguoi** | Người đi bộ | 1.65 m | Người | Tránh va chạm người đi cùng/ngược chiều; tự động **làm mờ mặt** bảo mật danh tính. |
| 2 | **Xe may** | Xe máy | 1.15 m | Xe cộ | Chướng ngại vật xuất hiện nhiều nhất trên vỉa hè và lòng đường Việt Nam. |
| 3 | **Xe hoi** | Xe ô tô | 1.55 m | Xe cộ | Ô tô đang lưu thông hoặc đỗ bên đường; cự ly cảnh báo ưu tiên cao. |
| 4 | **Xe dap** | Xe đạp | 1.10 m | Xe cộ | Phương tiện di chuyển chậm, dễ bất ngờ chuyển hướng cắt ngang. |
| 5 | **O ga** | Ổ gà / Hố sụt | 0.35 m | Mặt đường | Nguy hiểm tiềm ẩn dưới chân, giúp người khiếm thị phòng tránh trượt ngã. |
| 6 | **Cot** | Cột điện / Cột đèn | 2.50 m | Vật cản tĩnh | Cột điện, cột đèn chiếu sáng chiếm lối đi trên vỉa hè. |
| 7 | **Cay coi** | Cây cối / Thân cây | 2.50 m | Vật cản tĩnh | Gốc cây hoặc cành cây chìa thấp va đập vào vùng đầu và vai. |
| 8 | **Bang hieu** | Bảng hiệu quảng cáo | 1.20 m | Vật cản tĩnh | Biển hiệu quán xá, biển quảng cáo lấn chiếm vỉa hè. |
| 9 | **Thung rac** | Thùng rác | 0.90 m | Vật cản tĩnh | Thùng rác công cộng hoặc rác sinh hoạt đặt trước cửa nhà. |
| 10 | **Den giao thong** | Đèn tín hiệu | 2.80 m | Vật cản tĩnh | Cột đèn tín hiệu tại ngã ba/ngã tư, hỗ trợ định vị giao lộ. |

### 2. Bộ 80 lớp quốc tế COCO (Mô hình `gpModel.pt` / `yolov8n.pt`):
Hỗ trợ nhận diện bổ sung: con chó, con mèo, ghế đá công viên, ba lô, xe buýt, xe tải, v.v.

---

## ♿ TÍNH NĂNG TRỢ NĂNG DÀNH CHO NGƯỜI KHIẾM THỊ (WCAG 2.1 AAA)

1. **Điều khiển âm lượng loa chuẩn Laptop (Windows 11 / macOS)**:
   - Thanh trượt mượt mà $0\% - 100\%$ tích hợp trên header web.
   - Nút bật/tắt tiếng (Mute/Unmute) ghi nhớ mức âm lượng trước đó.
   - **Cuộn con lăn chuột trên nút loa** để tăng/giảm âm lượng $5\%$ tiện lợi.
   - Tự động phát âm thanh tick phản hồi giúp người khiếm thị cảm nhận ngay độ to/nhỏ.
2. **Menu chỉnh tốc độ nói kiểu YouTube ($0.5\text{x} - 2.0\text{x}$)**:
   - Cho phép người khiếm thị nghe quen có thể tăng tốc độ nói để phản xạ nhanh hơn trên đường.
3. **Bảng phím tắt điều khiển nhanh không cần nhìn màn hình**:
   - `Space`: Bật / Tắt quét camera
   - `M`: Bật / Tắt tiếng loa (Mute / Unmute)
   - `]`: Tăng âm lượng loa ($+5\%$)
   - `[`: Giảm âm lượng loa ($-5\%$)
   - `S`: Đổi tốc độ giọng nói
   - `C`: Đổi camera trước / sau (trên điện thoại)
   - `R`: Đọc lại câu cảnh báo gần nhất
   - `V`: Bấm nghe thử phát âm giọng đọc tiếng Việt
   - `p`: Tạm dừng / tiếp tục video (trên bản Desktop)
   - `q` hoặc `ESC`: Thoát ứng dụng an toàn

---

## 📁 CẤU TRÚC TOÀN BỘ THƯ MỤC DỰ ÁN

```text
YOLO_Project\
├── models\
│   ├── best.pt                 # Trọng số YOLO huấn luyện riêng 10 lớp vật cản Việt Nam (44MB)
│   ├── gpModel.pt              # Trọng số YOLOv8n 80 lớp COCO (6.5MB)
│   └── yolo26n.pt              # Trọng số YOLO COCO mới nhất (5.5MB)
├── src\
│   ├── __init__.py
│   ├── detector.py             # Mô-đun AI nhận diện (ObjectDetector & ObstacleDetector)
│   ├── voice_assistant.py      # Mô-đun âm thanh đa luồng Desktop (TTS + Volume)
│   ├── voice_alert.py          # Mô-đun phát âm thanh Web
│   └── visualizer.py           # Vẽ Bounding Box, HUD, vạch làn, làm mờ mặt
├── templates\
│   └── index.html              # Giao diện Web AI chính (3 video cùng 1 hàng, slider âm lượng laptop)
├── static\
│   ├── css\style.css           # Giao diện & thanh trượt âm lượng laptop
│   └── js\app.js               # Điều khiển webcam, giọng nói tiếng Việt & radar tiệm cận
├── test_video.mp4              # Video mẫu đường phố 1
├── test_video1.mp4             # Video mẫu xe máy & người đi bộ 2
├── test_video2.mp4             # Video mẫu chướng ngại vật lề đường 3
├── test_image.jpg              # Ảnh mẫu kiểm tra nhận diện ngã tư đô thị
├── web_app.py                  # Máy chủ FastAPI điều khiển Giao diện Web AI chính
├── main.py                     # Ứng dụng Desktop OpenCV (hỗ trợ webcam, video, chỉnh âm lượng)
├── config.py                   # Cấu hình camera, tỷ lệ vật thể, ngưỡng khoảng cách
├── run_web.bat                 # File 1-click khởi chạy Giao diện Web AI chính
├── run_desktop.bat             # File 1-click khởi chạy ứng dụng Desktop OpenCV
├── requirements.txt            # Danh sách thư viện phụ thuộc
├── .gitignore                  # Bỏ qua các tệp tạm thời
└── README.md                   # Tài liệu toàn diện của đồ án
```

---

## 🛠 HƯỚNG DẪN CÀI ĐẶT TỪ ĐẦU (CHO MÁY MỚI)

Nếu sao chép mã nguồn sang máy tính mới chưa có môi trường:

```bash
# Bước 1: Mở Terminal tại thư mục dự án YOLO_Project
cd YOLO_Project

# Bước 2: Cài đặt toàn bộ thư viện cần thiết
pip install -r requirements.txt

# Bước 3: Khởi chạy dự án bằng file .bat hoặc câu lệnh
run_web.bat
```

---

## 🏆 KẾT LUẬN & ĐÓNG GÓP CỦA ĐỀ TÀI

Đồ án **BlindAssist AI** mang lại một giải pháp công nghệ toàn diện và nhân văn:
1. **Tính chính xác & thích ứng cao:** Tích hợp mô hình nhận diện dành riêng cho vật cản giao thông thực tế tại Việt Nam (ổ gà, cột điện, xe máy lấn chiếm vỉa hè).
2. **Trải nghiệm âm thanh thông minh:** Không chỉ dừng lại ở giọng đọc cảnh báo, hệ thống còn kết hợp âm thanh tiệm cận Radar Sonar phản xạ nhanh và thanh điều chỉnh âm lượng linh hoạt như laptop.
3. **Đạo đức AI & Bảo vệ quyền riêng tư:** Tự động phát hiện và làm mờ khuôn mặt người đi bộ, tuân thủ tiêu chuẩn phát triển AI có trách nhiệm.
4. **Khả năng triển khai linh hoạt:** Chạy mượt mà từ máy tính cá nhân, giao diện Web trên điện thoại di động đến các thiết bị phần cứng nhúng di động.
