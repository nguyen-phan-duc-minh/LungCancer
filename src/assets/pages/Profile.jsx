import React, { useEffect, useState } from "react";
import axios from "axios";
import Header from '../components/Header';
import Footer from '../components/Footer';

const Profile = () => {
    const [profile, setProfile] = useState({ username: "", email: "", role: "" });
    const [oldPassword, setOldPassword] = useState("");
    const [newPassword, setNewPassword] = useState("");
    const [confirmNewPassword, setConfirmNewPassword] = useState("");
    const [message, setMessage] = useState("");

    const token = localStorage.getItem("token");

    useEffect(() => {
        axios.get("http://localhost:5001/me", {
            headers: { Authorization: `Bearer ${token}` }
        }).then(res => {
            setProfile(res.data);
        }).catch(err => {
            console.error("Không thể lấy thông tin người dùng", err);
        });
    }, [token]);

    const handlePasswordChange = (e) => {
        e.preventDefault();

        if (newPassword !== confirmNewPassword) {
            setMessage("Mật khẩu mới và xác nhận mật khẩu không khớp.");
            return;
        }

        axios.post("http://localhost:5001/update-password", {
            old_password: oldPassword,
            new_password: newPassword
        }, {
            headers: { Authorization: `Bearer ${token}` }
        }).then(res => {
            setMessage(res.data.message);
            setOldPassword("");
            setNewPassword("");
            setConfirmNewPassword("");
        }).catch(err => {
            setMessage(err.response?.data?.message || "Đã có lỗi xảy ra");
        });
    };

    const maskEmail = (email) => {
        if (!email) return "";
        const [user, domain] = email.split("@");
        if (user.length <= 3) return email; // Nếu quá ngắn thì không ẩn
        const visible = user.slice(0, 3);
        return `${visible}***@${domain}`;
    };

    return (
        <div className="fade-in">
            <Header />
            <div className="Profile p-4 max-w-md mx-auto">
                <form onSubmit={handlePasswordChange} className="flex flex-col gap-2 mt-2">
                    <h2 className="text-xl font-bold mb-4">Thông tin tài khoản</h2>
                    <input
                        type="text"
                        placeholder="Username"
                        value={profile.username}
                        readOnly
                        className="bg-gray-100 cursor-not-allowed p-2 rounded"
                    />
                    <input
                        type="text"
                        placeholder="Email"
                        value={maskEmail(profile.email)}
                        readOnly
                        className="bg-gray-100 cursor-not-allowed p-2 rounded"
                    />
                    {/* <input
                        type="text"
                        placeholder="Role"
                        value={profile.role}
                        readOnly
                        className="bg-gray-100 cursor-not-allowed p-2 rounded"
                    /> */}
                    <input
                        type="password"
                        placeholder="Mật khẩu cũ"
                        value={oldPassword}
                        onChange={(e) => setOldPassword(e.target.value)}
                        required
                    />
                    <input
                        type="password"
                        placeholder="Mật khẩu mới"
                        value={newPassword}
                        onChange={(e) => setNewPassword(e.target.value)}
                        required
                    />
                    <input
                        type="password"
                        placeholder="Xác nhận mật khẩu mới"
                        value={confirmNewPassword}
                        onChange={(e) => setConfirmNewPassword(e.target.value)}
                        required
                    />
                    <button type="submit" className="bg-blue-500 text-white p-2 rounded">Cập nhật</button>
                </form>

                {message && <p className="mt-2 text-sm text-red-500">{message}</p>}
            </div>
            <Footer />
        </div>
    );
};

export default Profile;
