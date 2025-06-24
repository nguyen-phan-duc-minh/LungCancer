import React, { useState, useEffect } from "react";
import Header from '../components/Header';
import Footer from '../components/Footer';

const Support = () => {
    const [faqs, setFaqs] = useState([]);
    const [contacts, setContacts] = useState({ phones: [], emails: [], addresses: [] });
    const [openIndex, setOpenIndex] = useState(null);

    useEffect(() => {
        // Fetch FAQs từ Flask backend
        fetch("http://localhost:5001/supports")
            .then(res => res.json())
            .then(data => setFaqs(data))
            .catch(err => console.error("Lỗi tải FAQs:", err));

        // Fetch Contact từ Flask backend
        fetch("http://localhost:5001/contacts")
            .then(res => res.json())
            .then(data => setContacts(data))
            .catch(err => console.error("Lỗi tải liên hệ:", err));
    }, []);

    return (
        <div className="OtherPage">
            <Header />
            <div className="Support">
                <section className="support-container">
                    <h1 className="page-title">Hỗ Trợ & Liên Hệ</h1>

                    <div className="faq-section">
                        {faqs.map((faq, index) => (
                            <div className="faq-item" key={index}>
                                <button
                                    className="faq-question"
                                    onClick={() =>
                                        setOpenIndex(openIndex === index ? null : index)
                                    }
                                >
                                    {faq.question}
                                </button>
                                {openIndex === index && (
                                    <div className="faq-answer">{faq.answer}</div>
                                )}
                            </div>
                        ))}
                    </div>

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
                </section>
            </div>
            <Footer />
        </div>
    );
};

export default Support;
