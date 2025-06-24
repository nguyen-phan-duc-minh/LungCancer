import React, { useEffect, useState } from "react";
import Header from "../components/HeaderAdmin";
import Footer from "../components/Footer";
import { motion } from "framer-motion";

const AdminTokenRequests = () => {
    const [requests, setRequests] = useState([]);
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(true);
    const [successMessage, setSuccessMessage] = useState("");

    const fetchRequests = async () => {
        try {
            const token = localStorage.getItem("token");
            const res = await fetch("http://localhost:5001/admin/pending-token-requests", {
                method: "GET",
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${token}`,
                },
                credentials: "include",
            });

            const data = await res.json();
            if (res.ok) {
                setRequests(data || []);
            } else {
                setError(data.message || "Không thể tải danh sách.");
            }
        } catch (err) {
            setError("Lỗi kết nối tới máy chủ.");
        } finally {
            setLoading(false);
        }
    };

    const handleStatusChange = async (requestId, newStatus) => {
        try {
            const token = localStorage.getItem("token");
            const res = await fetch(`http://localhost:5001/admin/update-token-request/${requestId}`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${token}`,
                },
                body: JSON.stringify({ status: newStatus }),
            });

            const data = await res.json();
            if (res.ok) {
                setSuccessMessage("Cập nhật trạng thái thành công!");
                fetchRequests();
                setTimeout(() => setSuccessMessage(""), 3000);
            } else {
                alert(data.message || "Cập nhật thất bại");
            }
        } catch (err) {
            alert("Lỗi khi cập nhật trạng thái");
        }
    };

    useEffect(() => {
        fetchRequests();
    }, []);

    return (
        <div className="fade-in">
            <Header />
            <div className="AdminTokenRequests">
                <motion.h2 initial={{ opacity: 0 }} animate={{ opacity: 1 }}>Danh sách yêu cầu mua Token</motion.h2>

                {loading && <p>Đang tải...</p>}
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

                {!loading && !error && (
                     <>
                        <table className="token-table">
                            <thead>
                                <tr>
                                    <th>Email</th>
                                    <th>Số Token</th>
                                    <th>Tổng tiền</th>
                                    <th>Ngày Gửi</th>
                                    <th>Minh Chứng</th>
                                    <th>Trạng Thái</th>
                                </tr>
                            </thead>
                            <tbody>
                                {requests.length === 0 ? (
                                    <tr>
                                        <td colSpan="6" style={{ textAlign: "center" }}>Không có yêu cầu nào.</td>
                                    </tr>
                                ) : (
                                    requests.map((req, index) => (
                                        <tr key={index}>
                                            <td>{req.user_email}</td>
                                            <td>{req.amount}</td>
                                            <td>{req.amount * 1000} đ</td> 
                                            <td>{new Date(req.created_at).toLocaleString()}</td>
                                            <td>
                                                {req.image ? (
                                                    <img
                                                        src={`http://localhost:5001/uploads/${req.image}`}
                                                        alt="Ảnh xác thực"
                                                        style={{ width: "100px", height: "auto", borderRadius: "6px" }}
                                                    />
                                                ) : (
                                                    <em>Không có ảnh</em>
                                                )}
                                            </td>
                                            <td>
                                                <select
                                                    value={req.status}
                                                    onChange={(e) => handleStatusChange(req.id, e.target.value)}
                                                >
                                                    <option value="pending">Pending</option>
                                                    <option value="approved">Approved</option>
                                                    <option value="rejected">Rejected</option>
                                                </select>
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

export default AdminTokenRequests;
