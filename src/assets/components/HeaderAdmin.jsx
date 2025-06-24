import React, { useState } from "react";
import '../css/headerAdmin.css';
import { FaBars } from "react-icons/fa"; // npm install react-icons

const HeaderAdmin = () => {
    const [menuOpen, setMenuOpen] = useState(false);

    return (
        <div className="HeaderAdmin">
            <div className="menu-toggle" onClick={() => setMenuOpen(!menuOpen)}>
                <FaBars />
            </div>
            <ul className={menuOpen ? "open" : ""}>
                <li><a href="/">Trở Về Trang Chủ</a></li>
                <li><a href="/AdminTokenRequests">Danh sách yêu cầu</a></li>
                <li><a href="/ContactManagement">QL Liên Hệ</a></li>
                <li><a href="/EmployeeManagement">QL Nhân Viên</a></li>
                <li><a href="/SupportManagement">QL Hỗ Trợ</a></li>
                <li><a href="/UserManager">QL Người Dùng</a></li>
                <li><a href="/PaymentManagement">QL đơn hàng</a></li>
            </ul>
        </div>
    );
};

export default HeaderAdmin;
