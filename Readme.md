--------------------------------------------------------------------------------
Credit Risk Classification Using Machine Learning
Giới thiệu dự án
Dự án này là Bài tập lớn môn Khai phá dữ liệu (CO3029) tại trường Đại học Bách Khoa - ĐHQG TP.HCM
. Mục tiêu chính là xây dựng một hệ thống học máy có khả năng dự đoán xác suất khách hàng vỡ nợ (loan default) dựa trên dữ liệu lịch sử từ Lending Club
.
Bài toán được tiếp cận dưới dạng Phân loại nhị phân (Binary Classification)
. Kết quả của mô hình giúp các tổ chức tài chính tối ưu hóa quy trình xét duyệt khoản vay và giảm thiểu rủi ro tín dụng
.
Thành viên thực hiện
Đặng Vũ Anh Khoa (MSSV: 2311578) - Tiền xử lý, EDA, Viết báo cáo
.
Nguyễn Trần Đức Hoàng (MSSV: 2311064) - Modeling, Viết báo cáo, Slide
.
Giảng viên hướng dẫn: Đỗ Thanh Thái
.
Dữ liệu sử dụng
Nguồn: Lending Club Loan Data (Kaggle)
.
Quy mô: Hơn 600.000 bản ghi với 105 thuộc tính ban đầu
.
Các nhóm thông tin chính: Đặc điểm khoản vay, thông tin người vay, lịch sử tín dụng và mục đích vay vốn
.
🛠 Quy trình thực hiện (KDD Process)
Dự án tuân thủ quy trình khám phá tri thức chuẩn bao gồm các bước
:
Tiền xử lý dữ liệu (Data Preprocessing):
Loại bỏ các biến có tỷ lệ thiếu > 70% và các biến không liên quan
.
Kiểm soát rò rỉ dữ liệu (Data Leakage): Loại bỏ các biến chứa thông tin phát sinh sau khi khoản vay đã được giải ngân (như recoveries, total_rec_int, last_pymnt_d)
.
Xử lý giá trị thiếu bằng trung vị (Median) và yếu vị (Mode)
.
Phân tích khám phá (EDA): Trực quan hóa mối quan hệ giữa các hạng tín dụng (Grade), thu nhập, tỷ lệ nợ (DTI) với khả năng vỡ nợ
.
Xây dựng đặc trưng (Feature Engineering):
Ordinal Encoding: Cho các biến có thứ tự như grade, sub_grade
.
One-Hot Encoding: Cho các biến danh mục như home_ownership, purpose
.
Biến đổi Log: Áp dụng cho thu nhập (log_income) để giảm độ lệch của dữ liệu
.
Mô hình hóa và Kết quả
Dự án thực hiện so sánh giữa thuật toán truyền thống trong đề cương và công nghệ mới
:
Mô hình cơ sở (Baseline): Decision Tree (Cây quyết định) - Đạt ROC-AUC: 0.7162
.
Mô hình nâng cao: XGBoost (Gradient Boosting) - Đạt ROC-AUC: 0.7418 sau khi tinh chỉnh siêu tham số bằng RandomizedSearchCV
.
Nhận xét: XGBoost cho thấy khả năng phân loại vượt trội và tính ổn định cao hơn so với Cây quyết định truyền thống
. Đặc trưng quan trọng nhất ảnh hưởng đến rủi ro là lãi suất (int_rate)
.
Triển khai (Deployment)
Mô hình cuối cùng được triển khai dưới dạng Dashboard tương tác sử dụng thư viện Streamlit
. Dashboard cho phép người dùng nhập thông tin tài chính và nhận kết quả dự báo mức độ rủi ro cùng xác suất vỡ nợ ngay lập tức
.
📂 Cấu trúc thư mục
/data: Chứa dữ liệu (nếu dung lượng cho phép) hoặc link download.
/notebooks: File Jupyter Notebook chi tiết quá trình EDA và Feature.
/preprocessing: Tiền xử lý dữ liệu.
/models: File Jupyter Notebook chi tiết quá trình Training.
Dashboard.py: dashboard báo cáo
Readme.md
xgboost_dashboard_model.plk

Hướng dẫn chạy:
    streamlit run Dashboard.py
--------------------------------------------------------------------------------