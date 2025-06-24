import React, { useEffect, useState } from "react";
import Header from "../components/HeaderAdmin";
import Footer from "../components/Footer";
import { motion } from "framer-motion";
import * as XLSX from "xlsx";
import { saveAs } from "file-saver";

const UserManagement = () => {
    const [users, setUsers] = useState([]);
    const [editingUser, setEditingUser] = useState(null);
    const [form, setForm] = useState({ role: "", tokens: 0 });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");
    const [successMessage, setSuccessMessage] = useState("");
    const token = localStorage.getItem("token");

    const exportToExcel = () => {
        if (users.length === 0) {
            alert("Không có dữ liệu để xuất.");
            return;
        }

        const worksheetData = users.map((user) => ({
            ID: user.id,
            "Tên đăng nhập": user.username,
            Email: user.email,
            Role: user.role,
            Tokens: user.tokens,
        }));

        const worksheet = XLSX.utils.json_to_sheet(worksheetData);
        const workbook = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(workbook, worksheet, "Users");

        const excelBuffer = XLSX.write(workbook, { bookType: "xlsx", type: "array" });
        const fileData = new Blob([excelBuffer], { type: "application/octet-stream" });
        saveAs(fileData, "DanhSachNguoiDung.xlsx");
    };

    const fetchUsers = async () => {
        try {
            const res = await fetch("http://localhost:5001/api/users", {
                headers: { Authorization: `Bearer ${token}` },
            });
            const data = await res.json();
            if (res.ok) {
                setUsers(data);
                setError("");
            } else {
                setError(data.message || "Không thể tải danh sách người dùng.");
            }
        } catch {
            setError("Lỗi kết nối máy chủ.");
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchUsers();
    }, []);

    const handleEdit = (user) => {
        setEditingUser(user);
        setForm({ role: user.role, tokens: user.tokens });
    };

    const handleUpdate = async () => {
        try {
            const res = await fetch(`http://localhost:5001/api/users/${editingUser.id}`, {
                method: "PUT",
                headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${token}`,
                },
                body: JSON.stringify(form),
            });
            const data = await res.json();
            if (res.ok) {
                setSuccessMessage("Cập nhật người dùng thành công!");
                setEditingUser(null);
                fetchUsers();
                setTimeout(() => setSuccessMessage(""), 3000);
            } else {
                setError(data.message || "Không thể cập nhật.");
            }
        } catch {
            setError("Lỗi kết nối khi cập nhật.");
        }
    };

    const handleDelete = async (id) => {
        if (!window.confirm("Bạn có chắc muốn xoá người dùng này?")) return;
        try {
            const res = await fetch(`http://localhost:5001/api/users/${id}`, {
                method: "DELETE",
                headers: { Authorization: `Bearer ${token}` },
            });
            const data = await res.json();
            if (res.ok) {
                setSuccessMessage("Xoá người dùng thành công!");
                fetchUsers();
                setTimeout(() => setSuccessMessage(""), 3000);
            } else {
                setError(data.message || "Không thể xoá.");
            }
        } catch {
            setError("Lỗi kết nối khi xoá.");
        }
    };

    return (
        <div className="fade-in">
            <Header />
            <div className="AdminTokenRequests">
                <motion.h2 initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    Quản lý Người dùng
                </motion.h2>

                {loading && <p>Đang tải dữ liệu...</p>}
                {error && <p style={{ color: "red" }}>{error}</p>}
                {successMessage && (
                    <motion.div className="success-msg" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                        {successMessage}
                    </motion.div>
                )}

                {!loading && !error && (
                       <>
                        <div style={{ marginBottom: "1em", textAlign: "right" }}>
                            <button onClick={exportToExcel} style={{ padding: "0.5em 1em",fontSize:"14px", fontWeight:"bold", backgroundColor: "rgb(41,41,220)", color: "white", border: "none", borderRadius: "4px" }}>
                                Xuất Excel
                            </button>
                        </div>
                    <table className="token-table">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Tên đăng nhập</th>
                                <th>Email</th>
                                <th>Role</th>
                                <th>Tokens</th>
                                <th>Hành động</th>
                            </tr>
                        </thead>
                        <tbody>
                            {users.map((user) => (
                                <tr key={user.id}>
                                    <td>{user.id}</td>
                                    <td>{user.username}</td>
                                    <td>{user.email}</td>
                                    <td>{user.role}</td>
                                    <td>{user.tokens}</td>
                                    <td>
                                        <button className="btnAdmin" onClick={() => handleEdit(user)}>Sửa</button>
                                        <button className="btnAdmin" onClick={() => handleDelete(user.id)} style={{ marginLeft: 8, color: "red" }}>Xoá</button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                    </>
                )}

                {/* FORM CHỈNH SỬA */}
                {editingUser && (
                    <div className="edit-form">
                        <h3>Chỉnh sửa: {editingUser.username}</h3>
                        <label>Role:</label>
                        <select
                            value={form.role}
                            onChange={(e) => setForm({ ...form, role: e.target.value })}
                        >
                            <option value="user">user</option>
                            <option value="admin">admin</option>
                        </select>
                        <label>Tokens:</label>
                        <input
                            type="number"
                            value={form.tokens}
                            onChange={(e) => setForm({ ...form, tokens: parseInt(e.target.value) })}
                        />
                        <button onClick={handleUpdate}>Cập nhật</button>
                        <button onClick={() => setEditingUser(null)} style={{ marginLeft: 8 }}>
                            Hủy
                        </button>
                    </div>
                )}
            </div>
            <Footer />
        </div>
    );
};

export default UserManagement;
