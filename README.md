# Đớp Tool v2

Đây là bản tích hợp lại đúng luồng bạn yêu cầu:

1. **Mô hình FEM**
   - Giữ canvas HTML/JS của web app ngày 08/10/2026.
   - Kéo chuột để vẽ thanh.
   - Gối tựa, di chuyển nút, xoá.
   - Gán vật liệu + tiết diện.
   - Gán tải nút và UDL.
   - Solve FEM khung phẳng 2D.
   - Xem N/V/M và đối chiếu Vereshchagin trong giao diện gốc.

2. **Tối ưu biến phân**
   - Lấy N(x), M(x) từ FEM.
   - Dựng ứng viên liên tục A*(x) bằng Euler–Lagrange/KKT.
   - Dùng ứng viên này làm điểm khởi tạo.
   - Tối ưu hữu hạn chiều với FEM trong vòng lặp để ràng buộc chuyển vị được kiểm tra trực tiếp.

3. **KKT**
   - Kiểm tra primal feasibility.
   - Dual feasibility.
   - Stationarity.
   - Complementary slackness.

4. **Thuyết minh**
   - Xuất DOCX gồm mô hình, hàm mục tiêu, biến phân, kết quả tối ưu và KKT.

## Chạy

```bash
pip install -r requirements.txt
streamlit run dop_tool_app.py
```

## Lưu ý kỹ thuật

Bản này ưu tiên đúng kiến trúc nghiên cứu: FEM không bị thay thế bởi tối ưu hóa.

Ứng viên biến phân sử dụng:
J[A] = ∫ A(x) dx

và ràng buộc năng lượng dọc:
∫ N(x)^2/[E A(x)] dx <= C.

Euler–Lagrange cho trường hợp không xuất hiện A'(x):
∂F/∂A = 0

cho:
A*(x) = max(|N(x)|/σ_allow, |N(x)| sqrt(λ/E)).

Đối với uốn, M(x) được lấy từ FEM và được dùng trong thành phần độ cứng uốn. Sau đó bài toán hữu hạn chiều kiểm tra chuyển vị FEM trực tiếp.

### Bước phát triển tiếp theo

- Truss FEM riêng với DOF [ux, uy].
- Tối ưu A(x) thuần biến phân cho giàn.
- Section family thực: I/H, hộp, ống tròn, thép góc.
- Quan hệ A-I theo hình học, không tối ưu A và I độc lập.
- Buckling Euler và ổn định theo TCVN 5575:2024.
- Nhiều trường hợp tải.
- Sensitivity FEM: dU/dp = -K^-1(dK/dp)U.
- Tối ưu toàn khung với biến thiết kế liên tục theo từng phần tử/đoạn.
- Chuyển nghiệm liên tục sang tiết diện thép thương mại.
