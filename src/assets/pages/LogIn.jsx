import React, { useState } from "react";
import { motion } from "framer-motion";
import Header from '../components/Header';
import Footer from '../components/Footer';
import axios from "axios";

const baseURL = import.meta.env.VITE_API_URL;

const LogIn = () => {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [error, setError] = useState("");
    const [successMessage, setSuccessMessage] = useState("");

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError("");
        setSuccessMessage("");

        try {
            const res = await fetch(`${baseURL}/login`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email, password }),
            });

            const data = await res.json();

            if (res.ok) {
                localStorage.setItem("token", data.token);
                localStorage.setItem("username", data.username);
                localStorage.setItem("role", data.role);
                setSuccessMessage("Đăng nhập thành công! Đang chuyển hướng...");
                setTimeout(() => {
                    window.location.href = "/";
                }, 500);
            } else {
                setError(data.message || "Email hoặc mật khẩu không đúng.");
            }
        } catch (error) {
            console.error("Lỗi:", error);
            setError("Đã xảy ra lỗi máy chủ. Vui lòng thử lại sau.");
        }
    };

    return (
        <div className="fade-in">
            <div className="LogIn">
                <motion.form onSubmit={handleSubmit}>
                    <h2>Đăng nhập</h2>

                    {error && (
                        <motion.p
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            style={{fontSize:"15px", color: 'red',marginBottom:'1em' }}
                        >
                            {error}
                        </motion.p>
                    )}

                    {successMessage && (
                        <motion.div
                            initial={{ opacity: 0, y: -20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.5 }}
                            style={{fontSize:"15px", color: '#27823c',marginBottom:'1em' }}
                        >
                            {successMessage}
                        </motion.div>
                    )}

                    <input
                        type="email"
                        placeholder="Email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        required
                    />
                    <input
                        type="password"
                        placeholder="Mật khẩu"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        required
                    />
                    <a className="ForgotPass" href="/ForgotPass">Quên mật khẩu?</a>
                    <button type="submit" style={{ marginTop: '15px' }}>Đăng nhập</button>
                    <a className="RegisterNow" href="/Register">Chưa có tài khoản? <span>Đăng ký ngay</span></a>
                </motion.form>
            </div>
        </div>
    );
};

export default LogIn;
