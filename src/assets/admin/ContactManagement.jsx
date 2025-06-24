import React, { useEffect, useState } from "react";
import Header from "../components/HeaderAdmin";
import Footer from "../components/Footer";
import { motion } from "framer-motion";
import * as XLSX from "xlsx";
import { saveAs } from "file-saver";

const ContactManagement = () => {
    const [contacts, setContacts] = useState([]);
    const [form, setForm] = useState({ contact_type: "", value: "", id: null });
    const [isEdit, setIsEdit] = useState(false);
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(true);
    const [successMessage, setSuccessMessage] = useState("");

    const exportToExcel = () => {
        if (contacts.length === 0) {
            alert("Không có dữ liệu để xuất.");
            return;
        }

        const worksheetData = contacts.map((c) => ({
            ID: c.id,
            "Loại liên hệ": c.contact_type,
            "Giá trị": c.value,
        }));

        const worksheet = XLSX.utils.json_to_sheet(worksheetData);
        const workbook = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(workbook, worksheet, "Contacts");

        const excelBuffer = XLSX.write(workbook, {
            bookType: "xlsx",
            type: "array",
        });

        const fileData = new Blob([excelBuffer], {
            type: "application/octet-stream",
        });

        saveAs(fileData, "DanhSachLienHe.xlsx");
    };

    const fetchContacts = async () => {
        try {
            const token = localStorage.getItem("token");
            const res = await fetch("http://localhost:5001/api/contacts", {
                headers: {
                    "Authorization": `Bearer ${token}`,
                    "Content-Type": "application/json",
                },
            });
            const data = await res.json();
            if (res.ok) {
                setContacts(data || []);
                setError("");
            } else {
                setError(data.message || "Không thể tải dữ liệu liên hệ.");
            }
        } catch {
            setError("Lỗi kết nối tới máy chủ.");
        } finally {
            setLoading(false);
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const token = localStorage.getItem("token");
        const url = form.id
            ? `http://localhost:5001/api/contacts/${form.id}`
            : "http://localhost:5001/api/contacts";
        const method = form.id ? "PUT" : "POST";

        try {
            const res = await fetch(url, {
                method,
                headers: {
                    "Authorization": `Bearer ${token}`,
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    contact_type: form.contact_type,
                    value: form.value,
                }),
            });
            const data = await res.json();
            if (res.ok) {
                setSuccessMessage(form.id ? "Cập nhật thành công!" : "Thêm mới thành công!");
                setForm({ contact_type: "", value: "", id: null });
                setIsEdit(false);
                fetchContacts();
                setTimeout(() => setSuccessMessage(""), 3000);
            } else {
                setError(data.message || "Lỗi khi xử lý");
            }
        } catch {
            setError("Lỗi kết nối máy chủ.");
        }
    };

    const handleDelete = async (id) => {
        if (!window.confirm("Bạn có chắc muốn xóa?")) return;
        const token = localStorage.getItem("token");
        try {
            const res = await fetch(`http://localhost:5001/api/contacts/${id}`, {
                method: "DELETE",
                headers: {
                    "Authorization": `Bearer ${token}`,
                },
            });
            const data = await res.json();
            if (res.ok) {
                setSuccessMessage("Xóa liên hệ thành công!");
                fetchContacts();
                setTimeout(() => setSuccessMessage(""), 3000);
            } else {
                setError(data.message || "Xóa thất bại.");
            }
        } catch {
            setError("Lỗi khi xóa.");
        }
    };

    const handleEdit = (c) => {
        setForm(c);
        setIsEdit(true);
    };

    const handleCancel = () => {
        setForm({ contact_type: "", value: "", id: null });
        setIsEdit(false);
        setError("");
    };

    useEffect(() => {
        fetchContacts();
    }, []);

    return (
        <div className="fade-in">
            <Header />
            <div className="AdminTokenRequests">
                <motion.h2 initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    Quản lý Liên hệ
                </motion.h2>

                {loading && <p>Đang tải dữ liệu...</p>}
                {error && <p style={{ color: "red" }}>{error}</p>}

                {successMessage && (
                    <motion.div
                        className="success-msg"
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                    >
                        {successMessage}
                    </motion.div>
                )}

                {/* FORM */}
                <form onSubmit={handleSubmit} className="contact-form">
                    <select
                        value={form.contact_type}
                        onChange={(e) => setForm({ ...form, contact_type: e.target.value })}
                        required
                    >
                        <option value="">-- Chọn loại liên hệ --</option>
                        <option value="phone">Số điện thoại</option>
                        <option value="email">Email</option>
                        <option value="address">Địa chỉ</option>
                    </select>

                    <input
                        type="text"
                        placeholder="Nhập giá trị"
                        value={form.value}
                        onChange={(e) => setForm({ ...form, value: e.target.value })}
                        required
                    />

                    <button type="submit">{isEdit ? "Cập nhật" : "Thêm mới"}</button>
                    {isEdit && (
                        <button type="button" onClick={handleCancel} style={{ marginLeft: "8px" }}>
                            Hủy
                        </button>
                    )}
                </form>

                {/* TABLE */}
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
                                <th>Loại</th>
                                <th>Giá trị</th>
                                <th>Hành động</th>
                            </tr>
                        </thead>
                        <tbody>
                            {contacts.length === 0 ? (
                                <tr>
                                    <td colSpan="4" style={{ textAlign: "center" }}>
                                        Không có liên hệ nào.
                                    </td>
                                </tr>
                            ) : (
                                contacts.map((c) => (
                                    <tr key={c.id}>
                                        <td>{c.id}</td>
                                        <td>{c.contact_type}</td>
                                        <td>{c.value}</td>
                                        <td>
                                            <button className="btnAdmin" onClick={() => handleEdit(c)}>Sửa</button>
                                            <button className="btnAdmin" onClick={() => handleDelete(c.id)}>Xóa</button>
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

export default ContactManagement;
