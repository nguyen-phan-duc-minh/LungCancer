import React, { useEffect, useState } from "react";
import Header from "../components/HeaderAdmin";
import Footer from "../components/Footer";
import { motion } from "framer-motion";
import * as XLSX from "xlsx";
import { saveAs } from "file-saver";

const PaymentManagement = () => {
    const [payments, setPayments] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");
    const [successMessage, setSuccessMessage] = useState("");

    const exportToExcel = () => {
        if (payments.length === 0) {
            alert("Không có dữ liệu để xuất.");
            return;
        }

        const worksheetData = payments.map((p) => ({
            "Email người dùng": p.user_email,
            "Số Token": p.amount,
            "Giá tiền": `${p.price.toLocaleString()} đ`,
            "Phương thức": p.method,
            "Thời gian": new Date(p.payment_time).toLocaleString(),
            "Trạng thái": p.status,
        }));

        const worksheet = XLSX.utils.json_to_sheet(worksheetData);
        const workbook = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(workbook, worksheet, "Payments");

        const excelBuffer = XLSX.write(workbook, {
            bookType: "xlsx",
            type: "array",
        });

        const fileData = new Blob([excelBuffer], {
            type: "application/octet-stream",
        });

        saveAs(fileData, "DanhSachThanhToan.xlsx");
    };

    const fetchPayments = async () => {
        try {
            const token = localStorage.getItem("token");
            const res = await fetch("http://localhost:5001/admin/payments", {
                headers: {
                    "Authorization": `Bearer ${token}`,
                    "Content-Type": "application/json",
                },
            });
            const data = await res.json();
            if (res.ok) {
                setPayments(data || []);
                setError("");
            } else {
                setError(data.message || "Không thể tải dữ liệu thanh toán.");
            }
        } catch {
            setError("Lỗi kết nối tới máy chủ.");
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchPayments();
    }, []);

    return (
        <div className="fade-in">
            <Header />
            <div className="AdminTokenRequests">
                <motion.h2 initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    Quản lý Thanh toán
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
                                <th>Email người dùng</th>
                                <th>Số Token</th>
                                <th>Giá tiền</th>
                                <th>Thời gian</th>
                                <th>Trạng thái</th>
                            </tr>
                        </thead>
                        <tbody>
                            {payments.length === 0 ? (
                                <tr>
                                    <td colSpan="6" style={{ textAlign: "center" }}>
                                        Không có giao dịch nào.
                                    </td>
                                </tr>
                            ) : (
                                payments.map((p) => (
                                    <tr key={p.id}>
                                        <td>{p.user_email}</td>
                                        <td>{p.amount}</td>
                                        <td>{p.price.toLocaleString()} đ</td>
                                        <td>{new Date(p.payment_time).toLocaleString()}</td>
                                        <td>{p.status}</td>
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

export default PaymentManagement;
