# ✅ CHECKLIST CÀI ĐẶT LẠI DỰ ÁN FEDERATED LEARNING

> In ra hoặc lưu file này để theo dõi tiến độ setup

---

## 🔧 PHẦN 1: CÀI ĐẶT CƠ BẢN

- [ ] **1.1** Tải và cài đặt Python 3.9+
      - Link: https://www.python.org/downloads/
      - ✅ Đã tick "Add Python to PATH"
      
- [ ] **1.2** Kiểm tra Python hoạt động
      ```powershell
      python --version
      ```
      Kết quả: `Python 3.9.x` hoặc cao hơn

- [ ] **1.3** Tải và cài đặt Git
      - Link: https://git-scm.com/download/win

- [ ] **1.4** Kiểm tra Git hoạt động
      ```powershell
      git --version
      ```
      Kết quả: `git version 2.x.x`

---

## 📥 PHẦN 2: CLONE DỰ ÁN

- [ ] **2.1** Mở PowerShell/Terminal

- [ ] **2.2** Di chuyển đến thư mục muốn lưu dự án
      ```powershell
      cd D:\
      ```

- [ ] **2.3** Clone repository
      ```powershell
      git clone https://github.com/hoangnguyenhtng/FederatedLearning.git
      ```
      ⏱️ Mất ~2-3 phút

- [ ] **2.4** Vào thư mục dự án
      ```powershell
      cd FederatedLearning
      ```

- [ ] **2.5** Kiểm tra files đã có
      ```powershell
      dir
      ```
      ✅ Thấy: src/, configs/, requirements.txt, etc.

---

## 🐍 PHẦN 3: SETUP MÔI TRƯỜNG PYTHON

- [ ] **3.1** Tạo môi trường ảo
      ```powershell
      python -m venv fed_rec_env
      ```
      ⏱️ Mất ~1-2 phút

- [ ] **3.2** Kích hoạt môi trường ảo
      ```powershell
      .\fed_rec_env\Scripts\Activate.ps1
      ```
      
      **Nếu gặp lỗi execution policy:**
      ```powershell
      Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
      ```
      Rồi chạy lại lệnh activate

- [ ] **3.3** Kiểm tra môi trường ảo đã active
      ✅ Thấy `(fed_rec_env)` ở đầu dòng lệnh

- [ ] **3.4** Nâng cấp pip
      ```powershell
      python -m pip install --upgrade pip
      ```

- [ ] **3.5** Cài đặt các thư viện
      ```powershell
      pip install -r requirements.txt
      ```
      ⏱️ Mất ~10-15 phút

- [ ] **3.6** Kiểm tra imports
      ```powershell
      python test_imports.py
      ```
      ✅ Không có lỗi = thành công!

---

## 📊 PHẦN 4: TẢI DỮ LIỆU

**Chọn 1 trong 3 options:**

- [ ] **Option A: Dataset nhỏ (KHUYẾN NGHỊ cho lần đầu)**
      ```powershell
      PowerShell -ExecutionPolicy Bypass -File download_amazon_data.ps1
      ```
      ⏱️ Mất ~10 phút

- [ ] **Option B: Dataset trung bình**
      ```powershell
      PowerShell -ExecutionPolicy Bypass -File download_amazon_multi_category.ps1
      ```
      ⏱️ Mất ~30 phút

- [ ] **Option C: Dataset đầy đủ**
      ```powershell
      PowerShell -ExecutionPolicy Bypass -File download_full_amazon_data.ps1
      ```
      ⏱️ Mất ~1-2 giờ

- [ ] **4.2** Kiểm tra dữ liệu đã tải
      ```powershell
      Test-Path data\raw\amazon_2023
      ```
      Kết quả: `True`

---

## ⚙️ PHẦN 5: XỬ LÝ DỮ LIỆU

- [ ] **5.1** Process dữ liệu Amazon
      ```powershell
      python src\data_generation\process_amazon_data.py
      ```
      ⏱️ Mất ~40-60 phút (dataset nhỏ)
      ⏱️ Mất ~2-3 giờ (dataset trung bình)
      ⏱️ Mất ~8-12 giờ (dataset đầy đủ - chạy overnight)

- [ ] **5.2** Kiểm tra dữ liệu đã process
      ```powershell
      Test-Path data\amazon_2023_processed\client_0\data.pkl
      ```
      Kết quả: `True`

- [ ] **5.3** Kiểm tra phân bổ dữ liệu
      ```powershell
      python check_data_distribution.py
      ```
      ✅ Xem thống kê về số lượng users, items, interactions

---

## 🏃 PHẦN 6: KIỂM TRA & CHẠY THỬ

- [ ] **6.1** Test dataloader
      ```powershell
      python test_dataloader.py
      ```
      ✅ Không có lỗi = sẵn sàng training!

- [ ] **6.2** Chạy training (TEST RUN - ít rounds)
      Mở file `configs\config.yaml` và sửa:
      ```yaml
      num_rounds: 5  # Thay vì 50
      ```
      
      Rồi chạy:
      ```powershell
      python src\training\federated_training_pipeline.py
      ```
      ⏱️ Mất ~5-10 phút

- [ ] **6.3** Kiểm tra kết quả test
      ```powershell
      Test-Path experiments\
      ```
      ✅ Có folder experiments với kết quả training

---

## 🚀 PHẦN 7: TRAINING THẬT

- [ ] **7.1** Đặt lại config (nếu đã test)
      Mở file `configs\config.yaml`:
      ```yaml
      num_rounds: 50  # Hoặc số rounds bạn muốn
      ```

- [ ] **7.2** Chạy full training
      ```powershell
      python src\training\federated_training_pipeline.py
      ```
      ⏱️ Mất ~30-45 phút (CPU)
      ⏱️ Mất ~15-20 phút (GPU)

- [ ] **7.3** Theo dõi quá trình training
      ✅ Xem accuracy tăng dần qua các rounds
      ✅ Loss giảm dần

- [ ] **7.4** Kiểm tra kết quả cuối cùng
      ```powershell
      # Xem file results
      cat experiments\fedper_multimodal_v1\results.json
      ```

---

## 📊 PHẦN 8: ĐÁNH GIÁ KẾT QUẢ

- [ ] **8.1** Kiểm tra metrics
      - Accuracy: [ __% ] (mục tiêu: 60-70%)
      - Loss: [ __ ] (mục tiêu: < 0.5)

- [ ] **8.2** Xem biểu đồ training
      Mở file: `experiments\fedper_multimodal_v1\training_history.png`

- [ ] **8.3** Chạy evaluation script (nếu có)
      ```powershell
      python src\training\evaluate_federated_model.py
      ```

---

## 💾 PHẦN 9: BACKUP & VERSION CONTROL

- [ ] **9.1** Backup kết quả training quan trọng
      Copy folder `experiments\` ra nơi khác

- [ ] **9.2** Commit code changes (nếu có sửa)
      ```powershell
      git add .
      git commit -m "Your changes description"
      git push origin main
      ```

- [ ] **9.3** Kiểm tra .gitignore hoạt động
      ```powershell
      git status
      ```
      ✅ KHÔNG thấy: fed_rec_env/, data/, experiments/

---

## 🎯 KẾT QUẢ MONG ĐỢI

### Với Amazon Dataset (10K samples):
- [x] Accuracy: 60-70%
- [x] Loss: ~0.5
- [x] Training time: 30-45 phút
- [x] Model học được patterns thực tế

### Với Amazon Dataset (Full):
- [x] Accuracy: 70-75%
- [x] Loss: ~0.3
- [x] Training time: 1-2 giờ
- [x] Kết quả tốt cho thesis/paper

---

## 🐛 LỖI THƯỜNG GẶP & GIẢI PHÁP

### Lỗi 1: Python không nhận diện
- [ ] Kiểm tra Python trong PATH
- [ ] Cài lại Python với "Add to PATH"

### Lỗi 2: Execution Policy
- [ ] Chạy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Lỗi 3: Out of Memory
- [ ] Giảm SAMPLE_SIZE trong `process_amazon_data.py`
- [ ] Đóng các ứng dụng khác

### Lỗi 4: Module not found
- [ ] Kiểm tra đã activate môi trường ảo chưa
- [ ] Chạy lại: `pip install -r requirements.txt`
- [ ] Chạy script từ thư mục gốc dự án

### Lỗi 5: CUDA/GPU issues
- [ ] Cài PyTorch CPU-only:
      ```powershell
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
      ```

---

## 📝 GHI CHÚ QUAN TRỌNG

**Thời gian setup:**
- Cài đặt cơ bản: ~30 phút
- Download + process data: ~1-3 giờ
- Training: ~30-45 phút
- **TỔNG: ~3-4 giờ**

**Dung lượng ổ cứng:**
- Code: ~50 MB
- Môi trường ảo: ~2 GB
- Data (nhỏ): ~500 MB
- Data (full): ~2-3 GB
- Models: ~100 MB
- **TỔNG: ~5-8 GB**

**Những thứ KHÔNG push lên Git:**
- ❌ fed_rec_env/
- ❌ data/
- ❌ experiments/
- ❌ __pycache__/
- ❌ *.pyc

---

## ✅ CHECKLIST HOÀN THÀNH

**Đánh dấu khi hoàn tất từng phần:**

- [ ] ✅ PHẦN 1: Cài đặt cơ bản (Python, Git)
- [ ] ✅ PHẦN 2: Clone dự án
- [ ] ✅ PHẦN 3: Setup môi trường Python
- [ ] ✅ PHẦN 4: Tải dữ liệu
- [ ] ✅ PHẦN 5: Xử lý dữ liệu
- [ ] ✅ PHẦN 6: Kiểm tra & test
- [ ] ✅ PHẦN 7: Training thật
- [ ] ✅ PHẦN 8: Đánh giá kết quả
- [ ] ✅ PHẦN 9: Backup & version control

---

## 🎉 HOÀN THÀNH!

Khi tất cả các mục đã được đánh dấu ✅, dự án của bạn đã sẵn sàng!

**Next steps:**
- Thử nghiệm với configs khác
- Explore notebooks
- Implement features mới
- Viết báo cáo/thesis

---

**Ngày hoàn thành**: _______________  
**Thời gian total**: _______________  
**Kết quả Accuracy**: ______________  
**Ghi chú**: _______________________

---

📖 **Tài liệu tham khảo:**
- HUONG_DAN_CAI_DAT_LAI.md (Chi tiết)
- SETUP_NHANH.txt (Quick guide)
- QUICK_START.md (Running guide)

🔗 **Repository**: https://github.com/hoangnguyenhtng/FederatedLearning.git

---

**Created**: 12/01/2026  
**Version**: 1.0  
**Status**: ✅ Ready to print
