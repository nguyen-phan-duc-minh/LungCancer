import React, { useEffect, useState } from "react";
import Header from "../components/HeaderAdmin";
import Footer from "../components/Footer";
import { motion } from "framer-motion";
import * as XLSX from "xlsx";
import { saveAs } from "file-saver";

const EmployeeManagement = () => {
    const [employees, setEmployees] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");
    const [form, setForm] = useState({ name: "", position: "", phone: "", email: "", image: "" });
    const [editId, setEditId] = useState(null);

    const token = localStorage.getItem("token");

    const exportToExcel = () => {
        if (employees.length === 0) {
            alert("Không có dữ liệu để xuất.");
            return;
        }

        const worksheetData = employees.map((emp) => ({
            ID: emp.id,
            "Họ tên": emp.name,
            "Chức vụ": emp.position,
            "Số điện thoại": emp.phone || "",
            "Email": emp.email || "",
            "Link ảnh": emp.image || "",
        }));

        const worksheet = XLSX.utils.json_to_sheet(worksheetData);
        const workbook = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(workbook, worksheet, "Employees");

        const excelBuffer = XLSX.write(workbook, {
            bookType: "xlsx",
            type: "array",
        });

        const fileData = new Blob([excelBuffer], {
            type: "application/octet-stream",
        });

        saveAs(fileData, "DanhSachNhanVien.xlsx");
    };

    const fetchEmployees = async () => {
        try {
            const res = await fetch("http://localhost:5001/employees", {
                headers: { Authorization: `Bearer ${token}` }
            });
            const data = await res.json();
            if (res.ok) setEmployees(data);
            else setError(data.message || "Không thể tải danh sách nhân viên.");
        } catch (err) {
            setError("Lỗi kết nối đến máy chủ.");
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => { fetchEmployees(); }, []);

    const handleChange = (e) => {
        setForm({ ...form, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const method = editId ? "PUT" : "POST";
        const url = editId ? `http://localhost:5001/employees/${editId}` : "http://localhost:5001/employees";
        try {
            const res = await fetch(url, {
                method,
                headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${token}`
                },
                body: JSON.stringify(form)
            });
            const data = await res.json();
            if (res.ok) {
                fetchEmployees();
                setForm({ name: "", position: "", phone: "", email: "", image: "" });
                setEditId(null);
            } else {
                alert(data.message || "Lỗi khi lưu.");
            }
        } catch (err) {
            alert("Lỗi kết nối đến server.");
        }
    };

    const handleEdit = (emp) => {
        setForm(emp);
        setEditId(emp.id);
    };

    const handleDelete = async (id) => {
        if (!window.confirm("Bạn có chắc muốn xoá nhân viên này?")) return;
        try {
            const res = await fetch(`http://localhost:5001/employees/${id}`, {
                method: "DELETE",
                headers: { Authorization: `Bearer ${token}` }
            });
            const data = await res.json();
            if (res.ok) fetchEmployees();
            else alert(data.message || "Không thể xoá.");
        } catch {
            alert("Lỗi khi xoá nhân viên.");
        }
    };

    return (
        <div className="fade-in">
            <Header />
            <div className="AdminTokenRequests">
                <motion.h2 initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    Quản lý nhân viên
                </motion.h2>

                {error && <p style={{ color: "red" }}>{error}</p>}

                <form onSubmit={handleSubmit} className="contact-form">
                    <input type="text" name="name" placeholder="Họ tên" value={form.name} onChange={handleChange} required />
                    <input type="text" name="position" placeholder="Chức vụ" value={form.position} onChange={handleChange} required />
                    <input type="text" name="phone" placeholder="Số điện thoại" value={form.phone} onChange={handleChange} />
                    <input type="email" name="email" placeholder="Email" value={form.email} onChange={handleChange} />
                    <input type="text" name="image" placeholder="Link ảnh (tuỳ chọn)" value={form.image} onChange={handleChange} />
                    <button type="submit">{editId ? "Cập nhật" : "Thêm mới"}</button>
                </form>

                {loading ? (
                    <p>Đang tải danh sách nhân viên...</p>
                ) : (
                    <>
                        <div style={{ marginBottom: "1em", textAlign: "right" }}>
                            <button onClick={exportToExcel} style={{ padding: "0.5em 1em",fontSize:"14px", fontWeight:"bold", backgroundColor: "rgb(41,41,220)", color: "white", border: "none", borderRadius: "4px" }}>
                                Xuất Excel
                            </button>
                        </div>
                    <table className="token-table">
                        <thead>
                            <tr>
                                <th>Ảnh</th>
                                <th>Họ tên</th>
                                <th>Chức vụ</th>
                                <th>SĐT</th>
                                <th>Email</th>
                                <th>Thao tác</th>
                            </tr>
                        </thead>
                        <tbody>
                            {employees.length === 0 ? (
                                <tr>
                                    <td colSpan="6">Chưa có nhân viên nào.</td>
                                </tr>
                            ) : (
                                employees.map((emp) => (
                                    <tr key={emp.id}>
                                        <td>
                                            {emp.image ? (
                                                <img src={`/src/assets/uploads/${emp.image}`} alt="avatar" style={{ width: "auto", maxHeight:"100px"}} />
                                            ) : (
                                                <em>Không có ảnh</em>
                                            )}
                                        </td>
                                        <td>{emp.name}</td>
                                        <td>{emp.position}</td>
                                        <td>{emp.phone}</td>
                                        <td>{emp.email}</td>
                                        <td>
                                            <button className="btnAdmin" onClick={() => handleEdit(emp)}>Sửa</button>
                                            <button className="btnAdmin" onClick={() => handleDelete(emp.id)} style={{ marginLeft: "8px", color: "red" }}>Xoá</button>
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                    </>
                )}
            </div>
            <Footer />
        </div>
    );
};

export default EmployeeManagement;
