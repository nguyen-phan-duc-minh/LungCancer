import React, { useState } from "react";
import { motion } from "framer-motion";
import Header from "../components/Header";
import Footer from "../components/Footer";

const BuyTokens = () => {
    const [amount, setAmount] = useState(10);
    const [message, setMessage] = useState("");
    const [error, setError] = useState("");
    const [image, setImage] = useState(null);

    const handleFileChange = (e) => {
        setImage(e.target.files[0]);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setMessage("");
        setError("");

        if (!amount || amount < 1) {
            setError("Vui lòng nhập số lượng token hợp lệ (>= 1).");
            return;
        }

        if (!image) {
            setError("Vui lòng chọn ảnh biên lai chuyển khoản.");
            return;
        }

        try {
            const token = localStorage.getItem("token");
            const formData = new FormData();
            formData.append("amount", amount);
            formData.append("image", image);

            const res = await fetch("http://localhost:5001/buy-tokens", {
                method: "POST",
                headers: {
                    Authorization: `Bearer ${token}`,
                },
                body: formData,
            });

            const data = await res.json();
            if (res.ok) {
                setMessage(data.message || "Yêu cầu mua token đã được gửi.");
                setAmount(10);
                setImage(null);
            } else {
                setError(data.message || "Đã xảy ra lỗi khi gửi yêu cầu.");
            }
        } catch (err) {
            setError("Không thể kết nối tới máy chủ.");
        }
    };

    return (
        <div className="fade-in">
            <Header />
            <div className="Register UpdatePass" style={{display:"flex", flexDirection:"unset"}}>
                <motion.form onSubmit={handleSubmit}>
                    <h2>Mua Token</h2>

                    {message && <p style={{ color: "green", fontWeight: "bold",margin:"0 0 1em", padding:"0 2em" ,textAlign:"center", fontSize:"15px" }}>{message}</p>}
                    {error && <p style={{ color: "red", fontWeight: "bold",margin:"0 0 1em", padding:"0 2em" ,textAlign:"center", fontSize:"15px"  }}>{error}</p>}

                    <input
                        type="number"
                        placeholder="Nhập số lượng token muốn mua"
                        value={amount}
                        onChange={(e) => setAmount(e.target.value)}
                        required
                        min={1}
                    />

                    <div style={{display:"flex", width:"80%", margin: "0.5em 0 1em", alignItems:"center", justifyContent:"space-between"}}>
                        <p style={{ fontWeight: "bold", fontStyle: "italic", color: "rgb(41,41,220)" }}>
                            1.000 đ = 1 token
                        </p>

                        <p style={{ fontWeight: "bold", color: "green" }}>
                            Tổng tiền: {Number(amount) * 1000} đ
                        </p>
                    </div>

                    <img style={{height:"140px", marginBottom:"0.5em"}} src="src/assets/images/Screenshot 2025-06-14 at 20.25.39.png" alt="" />
                    <p style={{marginBottom:"0.5em", fontWeight:"bold",fontSize:"18px"}}>Trần Minh Quang</p>

                    <input
                        type="file"
                        accept="image/*"
                        onChange={handleFileChange}
                    />

                    {/* {image && (
                    <img
                        src={URL.createObjectURL(image)}
                        alt="preview"
                        style={{ maxHeight: "100px", margin: "0.2em auto", display: "block", borderRadius: "8px" }}
                    />
                    )} */}

                    <button type="submit" style={{ marginTop: '15px' }}>
                        Gửi yêu cầu mua
                    </button>
                    <a className="RegisterNow" href="/">Quay lại trang chủ</a>
                </motion.form>
            </div>
            <Footer />
        </div>
    );
};

export default BuyTokens;
