import React, { useState } from "react";
import { motion } from "framer-motion";
import Header from "../components/Header";
import Footer from "../components/Footer";

const Register = () => {
    const [username, setUsername] = useState("");
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [confirmPassword, setConfirmPassword] = useState("");
    const [verificationCode, setVerificationCode] = useState("");

    const [codeSent, setCodeSent] = useState(false);
    const [successMessage, setSuccessMessage] = useState("");
    const [errorMessage, setErrorMessage] = useState("");

    const isStrongPassword = (password) => {
        const regex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[\W_]).{8,}$/;
        return regex.test(password);
    };

    const handleSendCode = async () => {
        setErrorMessage("");
        setSuccessMessage("");

        if (!email) {
            setErrorMessage("Vui lòng nhập email trước khi gửi mã xác thực.");
            return;
        }

        try {
            const response = await fetch("http://localhost:5001/send-verification-code", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email }),
                credentials: "include" 
            });

            const data = await response.json();

            if (response.ok) {
                setSuccessMessage("Mã xác thực đã được gửi về email.");
                setCodeSent(true);
            } else {
                setErrorMessage(data.message || "Không thể gửi mã xác thực.");
            }
        } catch (error) {
            console.error("Lỗi khi gửi mã:", error);
            setErrorMessage("Đã xảy ra lỗi khi gửi mã xác thực.");
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setErrorMessage("");
        setSuccessMessage("");

        if (password !== confirmPassword) {
            setErrorMessage("Mật khẩu xác nhận không khớp!");
            return;
        }

        if (!isStrongPassword(password)) {
            setErrorMessage("Mật khẩu phải có ít nhất 8 ký tự, gồm chữ hoa, chữ thường, số và ký tự đặc biệt!");
            return;
        }

        if (!verificationCode) {
            setErrorMessage("Vui lòng nhập mã xác thực.");
            return;
        }

        try {
            const response = await fetch("http://localhost:5001/register", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username, email, password, code: verificationCode }),
                credentials: "include",
            });

            const data = await response.json();

            if (response.ok) {
                setSuccessMessage("Đăng ký thành công! Đang chuyển sang trang đăng nhập...");
                setTimeout(() => {
                    window.location.href = "/LogIn";
                }, 2000);
            } else {
                setErrorMessage(data.message || "Đăng ký thất bại!");
            }
        } catch (error) {
            console.error("Lỗi khi đăng ký:", error);
            setErrorMessage("Đã xảy ra lỗi trong quá trình đăng ký.");
        }
    };

    return (
        <div className="fade-in">
            <div className="Register fade-in">
                <motion.form onSubmit={handleSubmit}>
                    <h2>Đăng ký tài khoản</h2>

                    {errorMessage && (
                        <motion.div  style={{fontSize:"15px", color: 'red',marginBottom:'1em' }}>
                            {errorMessage}
                        </motion.div>
                    )}
                    {successMessage && (
                        <motion.div style={{fontSize:"15px", color: '#27823c',marginBottom:'1em' }}>
                            {successMessage}
                        </motion.div>
                    )}

                    <input
                        type="text"
                        placeholder="Tên đăng nhập"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        required
                    />

                    <div style={{ display: "flex", alignItems: "center",width:"80%",justifyContent:"space-between", gap: "10px" }}>
                        <input
                            type="email"
                            placeholder="Email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                            style={{ width:"100%" }}
                        />
                        <button type="button" onClick={handleSendCode} style={{marginBottom:"1.5em", width: "30%"}}>
                            {codeSent ? "Gửi lại mã" : "Gửi mã"}
                        </button>
                    </div>

                    <input
                        type="text"
                        placeholder="Nhập mã xác thực"
                        value={verificationCode}
                        onChange={(e) => setVerificationCode(e.target.value)}
                        required
                    />

                    <input
                        type="password"
                        placeholder="Mật khẩu"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        required
                    />
                    <input
                        type="password"
                        placeholder="Xác nhận mật khẩu"
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                        required
                    />

                    <p style={{ fontSize: '13px', color: 'red', width: '90%', textAlign:"center" }}>
                        * Mật khẩu phải có ít nhất 8 ký tự, gồm chữ hoa, chữ thường, số và ký tự đặc biệt.
                    </p>

                    <button type="submit" style={{ marginTop: "15px" }}>Đăng ký</button>

                    <a className="RegisterNow" href="/LogIn">
                        Đã có tài khoản? <span>Đăng nhập ngay</span>
                    </a>
                </motion.form>
            </div>
        </div>
    );
};

export default Register;
