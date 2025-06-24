import React, { useEffect, useState } from "react";
import '../css/header.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
    faCircleInfo,
    faArrowUpFromBracket,
    faUserTie,
    faNotesMedical,
    faWandMagicSparkles,
    faBars,
    faXmark
} from '@fortawesome/free-solid-svg-icons';
import axios from 'axios';

const baseURL = import.meta.env.VITE_API_URL;

const Header = () => {
    const [isScrolled, setIsScrolled] = useState(false);
    const [username, setUsername] = useState(null);
    const [menuOpen, setMenuOpen] = useState(false);
    const [tokenCount, setTokenCount] = useState(null);
    const role = localStorage.getItem("role");

    // Handle scroll effect
    useEffect(() => {
        const handleScroll = () => {
            setIsScrolled(window.scrollY > 10);
        };
        window.addEventListener("scroll", handleScroll);

        // Fetch token count from server
        const fetchTokenCount = async () => {
            const jwt = localStorage.getItem("token");
            if (!jwt) return;

            try {
                const res = await axios.get(`${baseURL}/tokens`, {
                    headers: { Authorization: `Bearer ${jwt}` }
                });
                setTokenCount(res.data.tokens);
            } catch (error) {
                console.error("Lỗi khi lấy token:", error);
            }
        };

        fetchTokenCount(); // GỌI Ở ĐÂY

        return () => window.removeEventListener("scroll", handleScroll);
    }, []);

    // Lấy username từ localStorage
    useEffect(() => {
        const storedUser = localStorage.getItem("username");
        if (storedUser) {
            setUsername(storedUser);
        }
    }, []);

    // Đóng menu nếu resize lên desktop
    useEffect(() => {
        const handleResize = () => {
            if (window.innerWidth > 1023) {
                setMenuOpen(false);
            }
        };
        window.addEventListener("resize", handleResize);
        return () => window.removeEventListener("resize", handleResize);
    }, []);

    return (
        <header className={isScrolled ? "scrolled" : ""}>
            <div className="Container fade-down">
                <a href="/" className="Contain_img">
                    <img src="src/assets/images/logo.png" alt="logo" />
                </a>

                {/* Hamburger icon */}
                <button
                    className="menu-toggle"
                    onClick={() => setMenuOpen(!menuOpen)}
                    aria-label="Toggle menu"
                >
                    <FontAwesomeIcon icon={menuOpen ? faXmark : faBars} size="lg" />
                </button>

                {/* Navigation Menu */}
                <ul className={menuOpen ? "open" : ""}>
                    {role === "admin" && (
                        <li><a href="/AdminTokenRequests">Admin</a></li>
                    )}          
                    <li><a href="/"><FontAwesomeIcon icon={faArrowUpFromBracket} />Tải Ảnh Lên</a></li>
                    <li><a href="/Information"><FontAwesomeIcon icon={faWandMagicSparkles} />Về Ung Thư Phổi</a></li>
                    <li><a href="/History"><FontAwesomeIcon icon={faNotesMedical} />Lịch Sử</a></li>
                    <li><a href="/Support"><FontAwesomeIcon icon={faCircleInfo} />Hỗ Trợ & Liên Hệ</a></li>

                    {username ? (
                        <li className="UserInfor" style={{ display: "flex", alignItems: "center" }}>
                            <a href="/Profile">
                                <FontAwesomeIcon icon={faUserTie} />
                                &nbsp;Xin chào, {username.length > 10 ? username.slice(0, 10) + "..." : username}
                            </a>
                            <a href="/BuyTokens" style={{ marginLeft: "1em" }}>
                                Token: {tokenCount !== null ? tokenCount : "..."}
                            </a>
                            <button className="LogOut" onClick={() => {
                                localStorage.removeItem("token");
                                localStorage.removeItem("username");
                                localStorage.removeItem("role");
                                window.location.href = "/";
                            }}>
                                Đăng Xuất
                            </button>
                        </li>
                    ) : (
                        <li>
                            <a href="/LogIn">
                                <FontAwesomeIcon icon={faUserTie} />Đăng Nhập/Đăng Ký
                            </a>
                        </li>
                    )}
                </ul>
            </div>
        </header>
    );
};

export default Header;
