from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from PIL import Image
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta, datetime
from model.efficientnet_b1 import EfficientNetB1
from utils.transforms import preprocess_image
import re
import os
import smtplib
import random
import string
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr
import numpy as np
from sqlalchemy import inspect
from werkzeug.utils import secure_filename
import timm
from flask import session
from flask_session import Session

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Config
app.config['SESSION_TYPE'] = 'filesystem'  
app.config['SESSION_PERMANENT'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['JWT_SECRET_KEY'] = 'SWminh0918195615@gmail.com'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=1)
UPLOAD_FOLDER = 'src/assets/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.secret_key = 'SWminh0918195615'  # ví dụ: một chuỗi ngẫu nhiên dài và khó đoán
Session(app)
# Setup
db = SQLAlchemy(app)
jwt = JWTManager(app)
device = torch.device("cpu")

# Load model nhận diện CTScan vs NonCTScan
model_ctscan = timm.create_model("mobileone_s0", pretrained=False, num_classes=2)
model_ctscan.load_state_dict(torch.load("model/best_model.pth", map_location=device))
model_ctscan = model_ctscan.to(device)
model_ctscan.eval()

# Load model phân loại bệnh nếu là CTScan
model_disease = EfficientNetB1(num_classes=3).to(device)
checkpoint = torch.load("model/efficientnet_b1.pth", map_location=device)
model_disease.load_state_dict(checkpoint)
model_disease.eval()

# Class names
class_names_before_predict = ['CTScan', 'NonCTScan']
class_names = ['Bengin cases', 'Malignant cases', 'Normal cases']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_code(length=6):
    return ''.join(random.choices(string.digits, k=length))

def send_email(to_email, subject, message):
    sender = "SWminh0918195615@gmail.com"
    password = "mcyn lzvq vcqf hqzv"

    msg = MIMEText(message, "plain", "utf-8")
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"] = formataddr(("Lung Cancer AI", sender))
    msg["To"] = to_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, to_email, msg.as_string())

# Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(50), default='user')
    tokens = db.Column(db.Integer, default=10) 

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(150), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    result = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class Support(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String(255), nullable=False)
    answer = db.Column(db.Text, nullable=False)

class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    contact_type = db.Column(db.String(50))  # phone / email / address
    value = db.Column(db.String(255), nullable=False)

class Employee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    position = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(50))
    email = db.Column(db.String(150))
    image = db.Column(db.String(255))

class Payment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(150), nullable=False)
    amount = db.Column(db.Integer, nullable=False)  
    price = db.Column(db.Float, nullable=False)    
    payment_time = db.Column(db.DateTime, default=datetime.utcnow)
    method = db.Column(db.String(50), default="VNPAY") 
    status = db.Column(db.String(50), default="success") 

class TokenPurchaseRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(150), nullable=False)
    amount = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(50), default="pending")  # pending, approved, rejected
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    image = db.Column(db.String(255))

# Init DB
with app.app_context():
    # Tạo toàn bộ bảng nếu chưa có
    db.create_all()

    # Kiểm tra nếu bảng 'payment' chưa tồn tại thì tạo lại bảng
    inspector = inspect(db.engine)
    if 'payment' not in inspector.get_table_names():
        db.create_all()

    # Tạo tài khoản admin mặc định nếu chưa tồn tại
    admin_email = "admin@gmail.com"
    if not User.query.filter_by(email=admin_email).first():
        admin_user = User(
            username="admin",
            email=admin_email,
            password=generate_password_hash("admin"),
            role="admin"
        )
        db.session.add(admin_user)

    # Tạo dữ liệu Support mặc định nếu chưa có
    if Support.query.count() == 0:
        default_supports = [
            Support(question="Làm sao để đăng ký tài khoản?",
                    answer="Bạn điền form Đăng ký trên ứng dụng và xác thực email để hoàn tất quá trình tạo tài khoản."),
            Support(question="OncoFind hỗ trợ đa ngôn ngữ?",
                    answer="Hiện tại hỗ trợ tiếng Việt và tiếng Anh, phù hợp với đa số người dùng trong và ngoài nước."),
            Support(question="Dữ liệu bệnh nhân có được bảo mật?",
                    answer="Chúng tôi tuân thủ nghiêm ngặt các tiêu chuẩn bảo mật quốc tế như HIPAA và GDPR để bảo vệ thông tin người dùng."),
            Support(question="OncoFind dùng công nghệ gì để phát hiện ung thư phổi?",
                    answer="Ứng dụng sử dụng các mô hình học sâu như EfficientNet để phân tích ảnh X-quang, CT nhằm hỗ trợ phát hiện sớm dấu hiệu ung thư phổi."),
            Support(question="OncoFind có phải là công cụ chẩn đoán thay bác sĩ không?",
                    answer="Không. OncoFind là công cụ hỗ trợ, giúp bác sĩ phát hiện sớm dấu hiệu bất thường. Việc chẩn đoán chính thức vẫn do bác sĩ thực hiện."),
            Support(question="Tôi có thể dùng OncoFind trên điện thoại không?",
                    answer="Có. OncoFind hoạt động trên cả nền tảng web và ứng dụng di động (Android/iOS)."),
            Support(question="OncoFind có miễn phí không?",
                    answer="Bạn có thể sử dụng bản miễn phí với các chức năng cơ bản. Bản nâng cao yêu cầu đăng ký trả phí."),
            Support(question="Tôi có thể tải kết quả chẩn đoán không?",
                    answer="Có. Kết quả có thể tải về dưới dạng PDF hoặc hình ảnh để chia sẻ với bác sĩ hoặc lưu trữ cá nhân."),
        ]
        db.session.add_all(default_supports)

    # Tạo dữ liệu Contact mặc định nếu chưa có
    if Contact.query.count() == 0:
        default_contacts = [
            # Phones
            Contact(contact_type="phone", value="0901 234 567"),
            Contact(contact_type="phone", value="0902 345 678"),
            Contact(contact_type="phone", value="0903 456 789"),
            Contact(contact_type="phone", value="0904 567 890"),
            Contact(contact_type="phone", value="0905 678 901"),
            Contact(contact_type="phone", value="0906 789 012"),
            # Emails
            Contact(contact_type="email", value="support@oncofind.ai"),
            Contact(contact_type="email", value="contact@oncofind.ai"),
            Contact(contact_type="email", value="info@oncofind.ai"),
            Contact(contact_type="email", value="help@oncofind.ai"),
            Contact(contact_type="email", value="cskh@oncofind.vn"),
            # Addresses
            Contact(contact_type="address", value="Tầng 5, Tòa nhà ABC, Quận 3, TP. Hồ Chí Minh"),
            Contact(contact_type="address", value="Tầng 10, Tòa nhà XYZ, Quận 1, TP. Hồ Chí Minh"),
            Contact(contact_type="address", value="Số 21, Đường Lê Duẩn, Quận 1, TP. Hồ Chí Minh"),
            Contact(contact_type="address", value="92 Pasteur, Quận 1, TP. Hồ Chí Minh"),
            Contact(contact_type="address", value="Số 8, Tòa nhà TechTower, Quận Bình Thạnh, TP. Hồ Chí Minh"),
            Contact(contact_type="address", value="12 Nguyễn Văn Cừ, Quận 5, TP. Hồ Chí Minh"),
        ]
        db.session.add_all(default_contacts)

    # Tạo dữ liệu Employee mặc định nếu chưa có
    if Employee.query.count() == 0:
        employees = [
            Employee(name="Nguyễn Phan Đức Minh", position="Giám đốc kỹ thuật", phone="0934.1900.61", email="nckhqma@lungcancer.com", image="Minh.png"),
            Employee(name="Trần Minh Quang", position="Giám đốc kinh doanh", phone="0909.00 biết", email="quangcst@gmail.com", image="quang.jpg"),
            Employee(name="Phan Thiên An", position="Giám đốc sản xuất", phone="0909.00 biết", email="phanthienan@gmail.com", image="an.jpg"),
            Employee(name="Nguyễn Minh Mẫn", position="Tổng quản tiến độ dự án", phone="0909.00 biết", email="mannguyen@gmail.com", image="man.jpg"),
            Employee(name="Huỳnh Nguyễn Thanh Toàn", position="Tổng quản kỹ thuật", phone="0909.00 biết", email="toanhuynh@gmail.com", image="toan.jpg"),
            Employee(name="Lê Văn Quý", position="Nhân viên kỹ thuật", phone="0909.00 biết", email="quyvan@gmail.com", image="qvan.jpg"),
            Employee(name="Nguyễn Thành Quý", position="Nhân viên thực tập", phone="0909.00 biết", email="quyvan@gmail.com", image="quythanh.jpg"),
        ]
        db.session.add_all(employees)

    # Tạo tài khoản người dùng cho mỗi nhân viên nếu chưa có
    for emp in Employee.query.all():
        if not User.query.filter_by(email=emp.email).first():
            new_user = User(
                username=emp.name,
                email=emp.email,
                password=generate_password_hash(emp.email),  # Tạm thời dùng email làm mật khẩu
                role='user'
            )
            db.session.add(new_user)

    # Commit tất cả các thay đổi
    db.session.commit()

#-----------Contact--------------
# GET all contacts
@app.route('/api/contacts', methods=['GET'])
def get_all_contacts():
    contacts = Contact.query.all()
    return jsonify([{
        'id': c.id,
        'contact_type': c.contact_type,
        'value': c.value
    } for c in contacts])

# POST new contact
@app.route('/api/contacts', methods=['POST'])
def add_contact():
    data = request.get_json()
    contact = Contact(contact_type=data['contact_type'], value=data['value'])
    db.session.add(contact)
    db.session.commit()
    return jsonify({'message': 'Contact added successfully'})

# PUT update contact
@app.route('/api/contacts/<int:id>', methods=['PUT'])
def update_contact(id):
    data = request.get_json()
    contact = Contact.query.get_or_404(id)
    contact.contact_type = data['contact_type']
    contact.value = data['value']
    db.session.commit()
    return jsonify({'message': 'Contact updated successfully'})

# DELETE contact
@app.route('/api/contacts/<int:id>', methods=['DELETE'])
def delete_contact(id):
    contact = Contact.query.get_or_404(id)
    db.session.delete(contact)
    db.session.commit()
    return jsonify({'message': 'Contact deleted successfully'})

#-----------Employee--------------

@app.route("/employees", methods=["GET"])
def get_all_employees():
    employees = Employee.query.all()
    result = [
        {
            "id": e.id,
            "name": e.name,
            "position": e.position,
            "phone": e.phone,
            "email": e.email,
            "image": e.image
        }
        for e in employees
    ]
    return jsonify(result), 200

# CREATE employee
@app.route("/employees", methods=["POST"])
@jwt_required()
def create_employee():
    data = request.get_json()
    try:
        new_employee = Employee(
            name=data["name"],
            position=data["position"],
            phone=data.get("phone", ""),
            email=data.get("email", ""),
            image=data.get("image", "")
        )
        db.session.add(new_employee)
        db.session.commit()
        return jsonify({"message": "Thêm nhân viên thành công."}), 201
    except Exception as e:
        return jsonify({"message": "Lỗi khi thêm nhân viên.", "error": str(e)}), 400

# UPDATE employee
@app.route("/employees/<int:id>", methods=["PUT"])
@jwt_required()
def update_employee(id):
    employee = Employee.query.get(id)
    if not employee:
        return jsonify({"message": "Không tìm thấy nhân viên."}), 404

    data = request.get_json()
    try:
        employee.name = data["name"]
        employee.position = data["position"]
        employee.phone = data.get("phone", "")
        employee.email = data.get("email", "")
        employee.image = data.get("image", "")
        db.session.commit()
        return jsonify({"message": "Cập nhật nhân viên thành công."}), 200
    except Exception as e:
        return jsonify({"message": "Lỗi khi cập nhật nhân viên.", "error": str(e)}), 400

# DELETE employee
@app.route("/employees/<int:id>", methods=["DELETE"])
@jwt_required()
def delete_employee(id):
    employee = Employee.query.get(id)
    if not employee:
        return jsonify({"message": "Không tìm thấy nhân viên."}), 404
    try:
        db.session.delete(employee)
        db.session.commit()
        return jsonify({"message": "Xoá nhân viên thành công."}), 200
    except Exception as e:
        return jsonify({"message": "Lỗi khi xoá nhân viên.", "error": str(e)}), 400


#----------Payment-----------
@app.route("/admin/payments", methods=["GET"])
@jwt_required()
def get_all_payments():
    current_user_email = get_jwt_identity()
    user = User.query.filter_by(email=current_user_email).first()
    if user.role != "admin":
        return jsonify({'message': 'Bạn không có quyền'}), 403

    payments = Payment.query.order_by(Payment.payment_time.desc()).all()
    result = []
    for p in payments:
        result.append({
            "id": p.id,
            "user_email": p.user_email,
            "amount": p.amount,
            "price": p.price,
            "method": p.method,
            "payment_time": p.payment_time.isoformat(),
            "status": p.status
        })
    return jsonify(result), 200

#-----------Support--------------

@app.route("/api/supports", methods=["GET"])
@jwt_required()
def get_all_supports():
    supports = Support.query.all()
    return jsonify([{
        "id": s.id,
        "question": s.question,
        "answer": s.answer
    } for s in supports])

@app.route("/api/supports", methods=["POST"])
@jwt_required()
def create_support():
    data = request.get_json()
    new_support = Support(question=data["question"], answer=data["answer"])
    db.session.add(new_support)
    db.session.commit()
    return jsonify({"message": "Thêm thành công!"}), 201

@app.route("/api/supports/<int:support_id>", methods=["PUT"])
@jwt_required()
def update_support(support_id):
    support = Support.query.get_or_404(support_id)
    data = request.get_json()
    support.question = data["question"]
    support.answer = data["answer"]
    db.session.commit()
    return jsonify({"message": "Cập nhật thành công!"})

@app.route("/api/supports/<int:support_id>", methods=["DELETE"])
@jwt_required()
def delete_support(support_id):
    support = Support.query.get_or_404(support_id)
    db.session.delete(support)
    db.session.commit()
    return jsonify({"message": "Xoá thành công!"})


#------------User----------
@app.route("/api/users", methods=["GET"])
@jwt_required()
def get_users():
    identity = get_jwt_identity()  # string email
    user = User.query.filter_by(email=identity).first()
    if user.role != "admin":
        return jsonify({"msg": "Unauthorized"}), 403
    users = User.query.all()
    return jsonify([{
        "id": u.id,
        "username": u.username,
        "email": u.email,
        "role": u.role,
        "tokens": u.tokens
    } for u in users])

@app.route("/api/users/<int:user_id>", methods=["PUT"])
@jwt_required()
def update_user(user_id):
    identity = get_jwt_identity()
    print("Identity:", identity)
    if identity["role"] != "admin":
        return jsonify({"message": "Không có quyền"}), 403
    user = User.query.get_or_404(user_id)
    data = request.get_json()
    user.role = data.get("role", user.role)
    user.tokens = data.get("tokens", user.tokens)
    db.session.commit()
    return jsonify({"message": "Cập nhật người dùng thành công!"})

@app.route("/api/users/<int:user_id>", methods=["DELETE"])
@jwt_required()
def delete_user(user_id):
    identity = get_jwt_identity()
    if identity["role"] != "admin":
        return jsonify({"message": "Không có quyền"}), 403
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    return jsonify({"message": "Xoá người dùng thành công!"})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/buy-tokens", methods=["POST"])
@jwt_required()
def buy_tokens():
    token_amount = int(request.form.get("amount", 10))
    price_per_token = 1000
    total_price = token_amount * price_per_token
    current_user_email = get_jwt_identity()

    image = request.files.get('image')
    filename = None

    if image and allowed_file(image.filename):
        filename = secure_filename(f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{image.filename}")
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    request_entry = TokenPurchaseRequest(
        user_email=current_user_email,
        amount=token_amount,
        price=total_price,
        status="pending",
        image=filename
    )
    db.session.add(request_entry)
    db.session.commit()

    return jsonify({'message': 'Yêu cầu mua token đã được gửi, vui lòng chờ admin duyệt.'}), 200

@app.route("/admin/update-token-request/<int:request_id>", methods=["POST"])
@jwt_required()
def update_token_request_status(request_id):
    current_user_email = get_jwt_identity()
    user = User.query.filter_by(email=current_user_email).first()
    if user.role != "admin":
        return jsonify({'message': 'Bạn không có quyền'}), 403

    request_entry = TokenPurchaseRequest.query.get(request_id)
    if not request_entry:
        return jsonify({'message': 'Không tìm thấy yêu cầu'}), 404

    data = request.get_json()
    new_status = data.get("status")
    if new_status not in ["pending", "approved", "rejected"]:
        return jsonify({'message': 'Trạng thái không hợp lệ'}), 400

    # Nếu duyệt thành công thì thêm token + payment như trước
    if request_entry.status == "pending" and new_status == "approved":
        buyer = User.query.filter_by(email=request_entry.user_email).first()
        buyer.tokens += request_entry.amount
        payment = Payment(
            user_email=buyer.email,
            amount=request_entry.amount,
            price=request_entry.price,
            method="manual-admin",
            status="success"
        )
        db.session.add(payment)
    
    request_entry.status = new_status
    db.session.commit()

    return jsonify({'message': f'Trạng thái đã cập nhật thành {new_status}'}), 200

@app.route("/admin/approve-token-request/<int:request_id>", methods=["POST"])
@jwt_required()
def approve_token_request(request_id):
    current_user_email = get_jwt_identity()
    user = User.query.filter_by(email=current_user_email).first()

    if user.role != "admin":
        return jsonify({'message': 'Bạn không có quyền truy cập'}), 403

    request_entry = TokenPurchaseRequest.query.get(request_id)
    if not request_entry or request_entry.status != "pending":
        return jsonify({'message': 'Yêu cầu không hợp lệ hoặc đã được xử lý'}), 404

    # Cộng token cho người dùng
    buyer = User.query.filter_by(email=request_entry.user_email).first()
    buyer.tokens += request_entry.amount

    # Lưu thông tin thanh toán
    payment = Payment(
        user_email=buyer.email,
        amount=request_entry.amount,
        price=request_entry.price,
        method="manual-admin",
        status="success"
    )

    # Cập nhật trạng thái yêu cầu
    request_entry.status = "approved"

    db.session.add_all([buyer, payment, request_entry])
    db.session.commit()

    return jsonify({'message': f'Đã duyệt mua {request_entry.amount} token cho {buyer.email}'}), 200

@app.route("/admin/pending-token-requests", methods=["GET"])
@jwt_required()
def view_pending_requests():
    current_user_email = get_jwt_identity()
    user = User.query.filter_by(email=current_user_email).first()

    if user.role != "admin":
        return jsonify({'message': 'Bạn không có quyền truy cập'}), 403

    requests = TokenPurchaseRequest.query.filter_by(status="pending").all()
    result = [{
        'id': r.id,
        'user_email': r.user_email,
        'amount': r.amount,
        'price': r.price,
        'image': r.image,  # <-- thêm trường image
        'created_at': r.created_at.strftime('%Y-%m-%d %H:%M:%S')
    } for r in requests]

    return jsonify(result), 200

@app.route("/tokens", methods=["GET"])
@jwt_required()
def get_tokens():
    user_email = get_jwt_identity()
    user = User.query.filter_by(email=user_email).first()
    return jsonify({'tokens': user.tokens})

@app.route('/predict', methods=['POST'])
@jwt_required()
def predict():
    file = request.files['image']
    user_email = get_jwt_identity()
    user = User.query.filter_by(email=user_email).first()

    if user.tokens <= 0:
        return jsonify({
            'message': 'Bạn đã hết lượt dự đoán. Vui lòng mua thêm token.',
            'redirect_url': '/BuyTokens'
        }), 402

    try:
        filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        image = Image.open(filepath).convert('RGB')
        input_tensor = preprocess_image(image).to(device)

        # Bước 1: Dự đoán CT hay NonCT bằng model 1
        with torch.no_grad():
            outputs_ct = model_ctscan(input_tensor)  # model_ctscan là model đầu tiên
            _, pred_ct = torch.max(outputs_ct, 1)
            is_ct = (pred_ct.item() == 0)  # 0 = CTScan, 1 = NonCTScan

        if not is_ct:
            result = "NonCTScan"
        else:
            # Bước 2: Nếu là CT, phân loại bệnh
            with torch.no_grad():
                outputs_disease = model_disease(input_tensor)  # model_disease là EfficientNetB1
                _, pred_disease = torch.max(outputs_disease, 1)
                result = class_names[pred_disease.item()]  # Bệnh

        # Trừ token và lưu DB
        user.tokens -= 1
        db.session.add(user)
        new_entry = History(user_email=user_email, filename=filename, result=result)
        db.session.add(new_entry)
        db.session.commit()

        return jsonify({'prediction': result, 'remaining_tokens': user.tokens})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
@jwt_required()
def get_history():
    current_user = get_jwt_identity()
    histories = History.query.filter_by(user_email=current_user).order_by(History.id.desc()).all()

    results = [
        {
            'filename': h.filename,
            'result': h.result,
            'timestamp': h.timestamp.strftime('%Y-%m-%d %H:%M:%S') 
        }
        for h in histories
    ]
    return jsonify(results)

@app.route("/forgot-password", methods=["POST"])
def forgot_password():
    data = request.get_json()
    email = data.get("email")

    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"message": "Email không tồn tại"}), 404

    # Tạo mật khẩu mới
    new_pass = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    user.password = generate_password_hash(new_pass)
    db.session.commit()

    send_email(email, "Khôi phục mật khẩu", f"Mật khẩu mới của bạn là: {new_pass}")
    return jsonify({"message": "Đã gửi mật khẩu mới về email"}), 200

@app.route('/send-verification-code', methods=['POST'])
def send_verification_code():
    data = request.get_json()
    email = data['email']

    # Kiểm tra email đã tồn tại chưa
    if User.query.filter_by(email=email).first():
        return jsonify({'message': 'Email đã tồn tại'}), 400

    # Tạo mã xác thực và lưu vào session (hoặc Redis/cache)
    code = generate_code()
    session[email] = code  # Cần cấu hình SECRET_KEY cho Flask
    print("Session sau khi gửi mã:", dict(session))     
    # Gửi email
    send_email(
        to_email=email,
        subject="Mã xác thực đăng ký",
        message=f"Mã xác thực của bạn là: {code}"
    )

    return jsonify({'message': 'Đã gửi mã xác thực về email'}), 200

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()

    # Kiểm tra username và email
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'message': 'Tên đăng nhập đã tồn tại'}), 400
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'message': 'Email đã tồn tại'}), 400
    if not is_strong_password(data['password']):
        return jsonify({'message': 'Mật khẩu không đủ mạnh.'}), 400

    # Kiểm tra mã xác thực
    code = data.get('code')
    expected_code = session.get(data['email'])
    print("Session khi đăng ký:", dict(session))     # ✅ THÊM
    print("Client gửi mã:", code, " - Mã hệ thống:", expected_code)

    if not expected_code or code != expected_code:
        return jsonify({'message': 'Mã xác thực không hợp lệ'}), 400

    # Tạo tài khoản
    hashed_pw = generate_password_hash(data['password'])
    new_user = User(username=data['username'], email=data['email'], password=hashed_pw, role='user')
    db.session.add(new_user)
    db.session.commit()

    # Xóa mã xác thực
    session.pop(data['email'], None)

    return jsonify({'message': 'Đăng ký thành công'}), 201

@app.route("/employees", methods=["GET"])
def get_employees():
    employees = Employee.query.all()
    result = [
        {
            "id": e.id,
            "name": e.name,
            "position": e.position,
            "phone": e.phone,
            "email": e.email,
            "image": f"src/assets/uploads/{e.image}"  # hoặc đường dẫn tương đối tùy frontend
        }
        for e in employees
    ]
    return jsonify(result), 200

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(email=data['email']).first()

    if not user or not check_password_hash(user.password, data['password']):
        return jsonify({'message': 'Sai email hoặc mật khẩu'}), 401

    token = create_access_token(identity=user.email)
    return jsonify({'token': token, 'username': user.username, 'role': user.role}), 200

@app.route("/me", methods=["GET"])
@jwt_required()
def get_profile():
    current_user_email = get_jwt_identity()
    user = User.query.filter_by(email=current_user_email).first()
    if not user:
        return jsonify({'message': 'Không tìm thấy người dùng'}), 404

    return jsonify({
        'username': user.username,
        'email': user.email,
        'role': user.role
    }), 200

@app.route("/admin-only", methods=["GET"])
@jwt_required()
def admin_only():
    current_user_email = get_jwt_identity()
    user = User.query.filter_by(email=current_user_email).first()

    if user.role != "admin":
        return jsonify({"message": "Bạn không có quyền truy cập"}), 403

    return jsonify({"message": "Xin chào admin!"})

@app.route("/update-password", methods=["POST"])
@jwt_required()
def update_password():
    data = request.get_json()
    current_user_email = get_jwt_identity()
    user = User.query.filter_by(email=current_user_email).first()

    old_password = data.get("old_password")
    new_password = data.get("new_password")

    if not check_password_hash(user.password, old_password):
        return jsonify({'message': 'Mật khẩu cũ không đúng'}), 400

    if not is_strong_password(new_password):
        return jsonify({'message': 'Mật khẩu mới không đủ mạnh'}), 400

    user.password = generate_password_hash(new_password)
    db.session.commit()

    return jsonify({'message': 'Đổi mật khẩu thành công'}), 200

@app.route("/supports", methods=["GET"])
def get_supports():
    supports = Support.query.all()
    result = [{"question": s.question, "answer": s.answer} for s in supports]
    return jsonify(result), 200

@app.route("/contacts", methods=["GET"])
def get_contacts():
    phones = [c.value for c in Contact.query.filter_by(contact_type="phone")]
    emails = [c.value for c in Contact.query.filter_by(contact_type="email")]
    addresses = [c.value for c in Contact.query.filter_by(contact_type="address")]

    return jsonify({
        "phones": phones,
        "emails": emails,
        "addresses": addresses
    }), 200

def is_strong_password(password):
    return bool(re.match(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[\W_]).{8,}$', password))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
