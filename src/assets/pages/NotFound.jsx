import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import Header from "../components/Header";
import Footer from "../components/Footer";

const NotFound = () => {
    const navigate = useNavigate();
    const [countdown, setCountdown] = useState(2); // 2 giây

    useEffect(() => {
        const timer = setInterval(() => {
            setCountdown((prev) => {
                if (prev === 1) {
                    clearInterval(timer);
                    navigate("/");
                }
                return prev - 1;
            });
        }, 800);

        return () => clearInterval(timer);
    }, [navigate]);

    return (
        <div className="fade-in">
            <Header />
            <div className="LogIn">
                <motion.div>
                    <h2 style={{fontSize:"4em", textTransform:"uppercase", color:"#e74c3c"}}>404 - Không tìm thấy trang</h2>
                </motion.div>
            </div>
            <Footer />
        </div>
    );
};

export default NotFound;
