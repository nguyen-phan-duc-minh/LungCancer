import React, { useState, useRef, useEffect } from "react";
import Header from '../components/Header';
import Footer from '../components/Footer';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faImages, faAnglesDown, faGroupArrowsRotate, faUserTie } from '@fortawesome/free-solid-svg-icons'; // THÊM faUserTie
import axios from 'axios';
import { Canvas, useFrame } from '@react-three/fiber'; 
import { OrbitControls, Environment, useGLTF } from '@react-three/drei';

const baseURL = import.meta.env.VITE_API_URL;

function LungModel() {
  const { scene } = useGLTF('/models/realistic_human_lungs.glb');
  const ref = useRef();
  useFrame(() => {
    if (ref.current) {
      ref.current.rotation.y += 0.01;
    }
  });
  return <primitive ref={ref} object={scene} scale={13} />;
}

function HumanBody() {
  const { scene } = useGLTF('/models/ecorche_-_anatomy_study.glb');
  const ref = useRef();
  useFrame(() => {
    if (ref.current) {
      ref.current.rotation.y += 0.01;
    }
  });
  return <primitive ref={ref} object={scene} scale={200} />;
}

const Home = () => {
    const [selectedImages, setSelectedImages] = useState([]);
    const [previewURLs, setPreviewURLs] = useState([]);
    const [predictions, setPredictions] = useState([]);
    const [loading, setLoading] = useState(false);
    const uploadRef = useRef(null);
    const [contacts, setContacts] = useState({ phones: [], emails: [], addresses: [] });
    const [employees, setEmployees] = useState([]);

    // THÊM STATE username và token
    const [username, setUsername] = useState("");
    const [tokenCount, setTokenCount] = useState(null);

    useEffect(() => {
        // Lấy username và token từ localStorage
        const storedUsername = localStorage.getItem("username");
        if (storedUsername) {
            setUsername(storedUsername.length > 10 ? storedUsername.slice(0, 10) + "..." : storedUsername);
        }

        const token = localStorage.getItem("token");
        if (token) {
            axios.get(`${baseURL}/me`, {
                headers: {
                    Authorization: `Bearer ${token}`
                }
            })
            .then(res => setTokenCount(res.data.tokens))
            .catch(err => console.error("Không thể lấy số token:", err));
        }

        // Các API khác
        fetch("http://localhost:5001/contacts")
            .then(res => res.json())
            .then(data => setContacts(data))
            .catch(err => console.error("Lỗi tải liên hệ:", err));

        fetch("http://localhost:5001/employees")
            .then(res => res.json())
            .then(data => setEmployees(data))
            .catch(err => console.error("Lỗi tải nhân viên:", err));
    }, []);

    const handleUploadClick = () => {
        const token = localStorage.getItem("token");
        if (token) {
            uploadRef.current?.click();
        } else {
            alert("Vui lòng đăng nhập để sử dụng chức năng này!");
            window.location.href = "/LogIn";
        }
    };

    const handleImageUpload = async (e) => {
        const files = Array.from(e.target.files);
        setSelectedImages(files);
        setPreviewURLs(files.map(file => URL.createObjectURL(file)));

        const newPredictions = [];
        setLoading(true);
        const token = localStorage.getItem("token");

        for (const file of files) {
            const formData = new FormData();
            formData.append("image", file);

            try {
                const res = await axios.post(`${baseURL}/predict`, formData, {
                    headers: {
                        "Content-Type": "multipart/form-data",
                        "Authorization": `Bearer ${token}`
                    }
                });

                if (res.status === 200) {
                    newPredictions.push(res.data.prediction);
                    setTokenCount(res.data.remaining_tokens); // Cập nhật token
                }
            } catch (error) {
                if (error.response && error.response.status === 402) {
                    alert(error.response.data.message);
                    window.location.href = error.response.data.redirect_url || "/BuyTokens";
                    return;
                }
                console.error("Lỗi khi gửi ảnh:", error.response?.data || error.message);
                newPredictions.push("Lỗi");
            }
        }

        setPredictions(newPredictions);
        setLoading(false);
    };

    return (
        <div className="Total fade-in">
            <Header />
            {loading && (
                <div className="loading-overlay">
                    <div className="spinner"></div>
                    <p>Đang xử lý ảnh...</p>
                </div>
            )}
            <div className="Home">
                <div className="Section_1">
                    <div className="imageUpload"  onClick={handleUploadClick} >
                        <svg className="animated-border">
                            <rect
                                x="0" y="0" width="100%" height="100%"
                                rx="64" ry="64"
                                className="border-rect"
                            />
                        </svg>
                        <div className="uploadLabel" style={{ cursor: "pointer" }}>
                            <FontAwesomeIcon icon={faImages} className="uploadIcon" />
                            <p>Tải ảnh lên tại đây</p>
                        </div>
                        <input
                            type="file"
                            accept="image/*"
                            // capture="environment" // hoặc "user" cho camera trước
                            multiple
                            ref={uploadRef}
                            style={{ display: 'none' }}
                            onChange={handleImageUpload}
                        />
                    </div>

                    {previewURLs.length > 0 && (
                        <div className="imageResult">
                            <table className="result-table">
                                <thead>
                                    <tr>
                                        <th>Ảnh</th>
                                        <th>Dự đoán</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {previewURLs.map((url, index) => (
                                        <tr key={index}>
                                            <td>
                                                <img src={url} alt={`Ảnh ${index}`} width="100" />
                                            </td>
                                            <td>{predictions[index] || "Đang xử lý..."}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>
            </div>

            <div className="Section_2">
                <div className="Contain_model">
                    <Canvas camera={{ position: [0, 0, 5] }}>
                        <ambientLight intensity={0.8} />
                        <Environment preset="city" />
                        <LungModel />
                    </Canvas>
                </div>
                <div className="Contain">
                    <a href="/Information" className="Content">
                        <h2>Ung Thư Phổi</h2>
                        <p>
                            Ung thư phổi là nguyên nhân gây tử vong hàng đầu do ung thư, với khoảng <strong>2,5 triệu ca mắc</strong> và <strong>1,8 triệu ca tử vong mỗi năm</strong> trên toàn thế giới (GLOBOCAN 2024).
                        </p>
                        <p>
                            Tỷ lệ mắc ung thư phổi đã <strong>tăng gấp đôi</strong> trong 20 năm qua, chủ yếu do hút thuốc, ô nhiễm không khí và phát hiện muộn.
                        </p>
                        <p>
                            Tại Việt Nam, mỗi năm ghi nhận hơn <strong>26.000 ca mắc mới</strong> và gần <strong>24.000 ca tử vong</strong>, đứng thứ hai sau ung thư gan.
                        </p>
                    </a>
                </div>
            </div>

            <div className="Section_2_overall">
                <div className="inner left">
                    <FontAwesomeIcon icon={faGroupArrowsRotate} />
                    <p>Xoay ảnh để trải nghiệm</p>
                </div>
                <div className="Contain_model">
                    <Canvas camera={{ position: [0, 0, 5] }}>
                        <ambientLight intensity={0.8} />
                        <Environment preset="city" />
                        <HumanBody/>
                        <OrbitControls/>
                    </Canvas>
                </div>
                <div className="inner right">
                    <FontAwesomeIcon icon={faAnglesDown} />
                    <p>Vui lòng lướt xuống ngoài đây</p>
                </div>
            </div>

            <div className="Section_3" style={{marginTop:"4em"}}>
                <h2>Đội Ngũ Nhân Viên</h2>
                <ul className="Container">
                    {employees.map((emp, index) => (
                        <li key={index}>
                            <a href="">
                                <div className="Contain_img">
                                    <img src={`/src/assets/uploads/${emp.image}`} alt={emp.name} />
                                </div>
                                <div className="infor">
                                    <h3>{emp.name}</h3>
                                    <p>{emp.position}</p>
                                    <p>SĐT: {emp.phone}</p>
                                    <p>Email: {emp.email}</p>
                                </div>
                            </a>
                        </li>
                    ))}
                </ul>
            </div>

            <div className="Section_4">
                <div className="Container">
                     <div className="faq-contact">
                        <h3>Liên hệ</h3>
                        
                        {contacts.phones.map((phone, i) => (
                            <p key={`phone-${i}`}><strong>📞 Số điện thoại:</strong> {phone}</p>
                        ))}
                        {contacts.emails.map((email, i) => (
                            <p key={`email-${i}`}><strong>📧 Email:</strong> {email}</p>
                        ))}
                        {contacts.addresses.map((addr, i) => (
                            <p key={`addr-${i}`}><strong>📍 Địa chỉ:</strong> {addr}</p>
                        ))}
                    </div>
                </div>
            </div>
            <Footer />
        </div>
    );
};

export default Home;
