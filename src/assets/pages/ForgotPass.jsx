import React, { useState } from "react";
import { motion } from "framer-motion";
import Header from '../components/Header';
import Footer from '../components/Footer';

const ForgotPass = () => {
    const [email, setEmail] = useState("");
    const [message, setMessage] = useState("");
    const [error, setError] = useState("");

    const handleSubmit = async (e) => {
        e.preventDefault();
        setMessage("");
        setError("");

        try {
            const res = await fetch("http://localhost:5001/forgot-password", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email }),
            });

            const data = await res.json();
            if (res.ok) {
                setMessage(data.message || "Đã gửi email thành công!");
            } else {
                setError(data.message || "Có lỗi xảy ra.");
            }
        } catch (err) {
            setError("Không thể kết nối tới máy chủ.");
        }
    };

    return (
        <div className="fade-in">
            <Header />
            <div className="LogIn">
                <motion.form
                    onSubmit={handleSubmit}
                >
                    <h2>Quên mật khẩu</h2>

                    {message && <p style={{ color: "green", fontWeight: "bold" }}>{message}</p>}
                    {error && <p style={{ color: "red", fontWeight: "bold" }}>{error}</p>}

                    <input
                        type="email"
                        placeholder="Nhập email của bạn"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        required
                    />

                    <button type="submit" style={{ marginTop: '15px' }}>Gửi mật khẩu mới</button>
                    <a className="RegisterNow" href="/LogIn">Quay lại đăng nhập</a>
                </motion.form>
            </div>
            <Footer />
        </div>
    );
};

export default ForgotPass;
