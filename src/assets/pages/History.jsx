import React, { useEffect, useState } from "react";
import Header from '../components/Header';
import Footer from '../components/Footer';
import axios from 'axios';

const baseURL = import.meta.env.VITE_API_URL;

const formatDateVN = (date) => {
  const d = date.getDate().toString().padStart(2, '0');
  const m = (date.getMonth() + 1).toString().padStart(2, '0');
  const y = date.getFullYear();
  return `${d}/${m}/${y}`;
};

const History = () => {
    const [groupedHistory, setGroupedHistory] = useState({});
    const [selectedImage, setSelectedImage] = useState(null);

    useEffect(() => {
        const fetchHistory = async () => {
            try {
                const token = localStorage.getItem("token");
                const res = await axios.get(`${baseURL}/history`, {
                    headers: {
                        Authorization: `Bearer ${token}`
                    }
                });

                const grouped = res.data.reduce((acc, item) => {
                    const date = new Date(item.timestamp);
                    // Chuyển timestamp sang UTC+7
                    const utc7Date = new Date(date.getTime() + 7 * 60 * 60 * 1000);
                    const dayStr = formatDateVN(utc7Date);  // "dd/mm/yyyy"
                    if (!acc[dayStr]) acc[dayStr] = [];
                    acc[dayStr].push(item);
                    return acc;
                }, {});

                // Sắp xếp ngày mới nhất lên đầu
                const sorted = Object.fromEntries(
                    Object.entries(grouped).sort((a, b) => {
                        // Chuyển "dd/mm/yyyy" -> "yyyy-mm-dd" để tạo Date chuẩn
                        const da = new Date(a[0].split('/').reverse().join('-'));
                        const db = new Date(b[0].split('/').reverse().join('-'));
                        return db - da; // ngày mới hơn đứng trước
                    })
                );

                setGroupedHistory(sorted);
            } catch (error) {
                console.error("Lỗi khi tải lịch sử:", error.response?.data || error.message);
            }
        };

        fetchHistory();
    }, []);

    return (
        <div className="OtherPage">
            <Header />
            <div className="History">
                <h2>Lịch sử dự đoán</h2>
                {Object.keys(groupedHistory).length > 0 ? (
                    Object.entries(groupedHistory).map(([date, items]) => (
                        <div className="ScrollClass" key={date} style={{ width:"100% !important", marginBottom: "20px" }}>
                            <h3>Ngày {date}</h3>
                            <ul>
                                {items.map((item, index) => {
                                    const itemDate = new Date(item.timestamp);
                                    const utc7Time = new Date(itemDate.getTime() + 7 * 60 * 60 * 1000);
                                    const timeStr = utc7Time.toLocaleTimeString('vi-VN');
                                    return (
                                        <li key={index} style={{ marginBottom: "10px" }}>
                                            <div className="Contain_img">
                                                <img
                                                    src={`http://localhost:5001/uploads/${item.filename}`}
                                                    alt="Ảnh dự đoán"
                                                    onClick={() => setSelectedImage(`http://localhost:5001/uploads/${item.filename}`)}
                                                    style={{ cursor: 'pointer', maxWidth: '150px', maxHeight: '150px' }}
                                                />
                                            </div>
                                            <p>{item.result}</p>
                                            <p>{timeStr}</p>
                                        </li>
                                    );
                                })}
                            </ul>
                        </div>
                    ))
                ) : (
                    <p>Chưa có dự đoán nào</p>
                )}
            </div>

            {selectedImage && (
                <div
                    className="fullscreen-overlay"
                    onClick={() => setSelectedImage(null)}
                    style={{
                        position: "fixed",
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        backgroundColor: "rgba(0, 0, 0, 0.85)",
                        display: "flex",
                        justifyContent: "center",
                        alignItems: "center",
                        zIndex: 9999,
                        cursor: "zoom-out"
                    }}
                >
                    <img
                        src={selectedImage}
                        alt="Ảnh phóng to"
                        style={{ maxWidth: "90%", maxHeight: "90%", objectFit: "contain" }}
                    />
                </div>
            )}
        </div>
    );
};

export default History;
