import React, { useEffect, useState } from "react";
import Header from "../components/HeaderAdmin";
import Footer from "../components/Footer";
import { motion } from "framer-motion";
import * as XLSX from "xlsx";
import { saveAs } from "file-saver";

const SupportManagement = () => {
    const [supports, setSupports] = useState([]);
    const [form, setForm] = useState({ question: "", answer: "", id: null });
    const [isEdit, setIsEdit] = useState(false);
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(true);
    const [successMessage, setSuccessMessage] = useState("");
    const token = localStorage.getItem("token");

    const exportToExcel = () => {
        if (supports.length === 0) {
            alert("Không có dữ liệu để xuất.");
            return;
        }

        const worksheetData = supports.map((s) => ({
            "ID": s.id,
            "Câu hỏi": s.question,
            "Câu trả lời": s.answer,
        }));

        const worksheet = XLSX.utils.json_to_sheet(worksheetData);
        const workbook = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(workbook, worksheet, "Supports");

        const excelBuffer = XLSX.write(workbook, { bookType: "xlsx", type: "array" });
        const fileData = new Blob([excelBuffer], { type: "application/octet-stream" });
        saveAs(fileData, "DanhSachCauHoiHoTro.xlsx");
    };

    const fetchSupports = async () => {
        try {
            const res = await fetch("http://localhost:5001/api/supports", {
                headers: { Authorization: `Bearer ${token}` }
            });
            const data = await res.json();
            if (res.ok) {
                setSupports(data);
                setError("");
            } else {
                setError(data.message || "Không thể tải dữ liệu hỗ trợ.");
            }
        } catch {
            setError("Lỗi kết nối tới máy chủ.");
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => { fetchSupports(); }, []);

    const handleSubmit = async (e) => {
        e.preventDefault();
        const url = form.id
            ? `http://localhost:5001/api/supports/${form.id}`
            : "http://localhost:5001/api/supports";
        const method = form.id ? "PUT" : "POST";

        try {
            const res = await fetch(url, {
                method,
                headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${token}`
                },
                body: JSON.stringify({
                    question: form.question,
                    answer: form.answer
                })
            });
            const data = await res.json();
            if (res.ok) {
                setSuccessMessage(form.id ? "Cập nhật thành công!" : "Thêm mới thành công!");
                setForm({ question: "", answer: "", id: null });
                setIsEdit(false);
                fetchSupports();
                setTimeout(() => setSuccessMessage(""), 3000);
            } else {
                setError(data.message || "Lỗi xử lý");
            }
        } catch {
            setError("Lỗi kết nối.");
        }
    };

    const handleEdit = (item) => {
        setForm(item);
        setIsEdit(true);
    };

    const handleDelete = async (id) => {
        if (!window.confirm("Bạn có chắc muốn xoá câu hỏi này?")) return;
        try {
            const res = await fetch(`http://localhost:5001/api/supports/${id}`, {
                method: "DELETE",
                headers: { Authorization: `Bearer ${token}` }
            });
            const data = await res.json();
            if (res.ok) {
                fetchSupports();
                setSuccessMessage("Xoá thành công!");
                setTimeout(() => setSuccessMessage(""), 3000);
            } else {
                setError(data.message || "Không thể xoá.");
            }
        } catch {
            setError("Lỗi xoá câu hỏi.");
        }
    };

    const handleCancel = () => {
        setForm({ question: "", answer: "", id: null });
        setIsEdit(false);
        setError("");
    };

    return (
        <div className="fade-in">
            <Header />
            <div className="AdminTokenRequests">
                <motion.h2 initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    Quản lý Câu hỏi Hỗ trợ
                </motion.h2>

                {loading && <p>Đang tải dữ liệu...</p>}
                {error && <p style={{ color: "red" }}>{error}</p>}

                {successMessage && (
                    <motion.div className="success-msg" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                        {successMessage}
                    </motion.div>
                )}

                {/* FORM */}
                <form onSubmit={handleSubmit} className="contact-form">
                    <input
                        type="text"
                        placeholder="Câu hỏi"
                        value={form.question}
                        onChange={(e) => setForm({ ...form, question: e.target.value })}
                        required
                    />
                    <textarea
                        placeholder="Câu trả lời"
                        value={form.answer}
                        onChange={(e) => setForm({ ...form, answer: e.target.value })}
                        required
                        rows={3}
                        style={{height: "50px",width:"40vw", padding: "8px"}}
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
                                <th>Câu hỏi</th>
                                <th>Câu trả lời</th>
                                <th>Hành động</th>
                            </tr>
                        </thead>
                        <tbody>
                            {supports.length === 0 ? (
                                <tr>
                                    <td colSpan="4" style={{ textAlign: "center" }}>Không có dữ liệu.</td>
                                </tr>
                            ) : (
                                supports.map((s) => (
                                    <tr key={s.id}>
                                        <td>{s.id}</td>
                                        <td>{s.question}</td>
                                        <td>{s.answer}</td>
                                        <td>
                                            <button className="btnAdmin" onClick={() => handleEdit(s)}>Sửa</button>
                                            <button className="btnAdmin" onClick={() => handleDelete(s.id)} style={{ color: "red" }}>Xoá</button>
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

export default SupportManagement;
