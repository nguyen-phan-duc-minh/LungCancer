import React, { useEffect, useState } from "react";
import '../css/footer.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faAt, faMobileScreenButton } from '@fortawesome/free-solid-svg-icons';
import { faFacebookMessenger, faFacebookF, faInstagram, faTelegram } from '@fortawesome/free-brands-svg-icons';

const Footer = () => {
    const [showFooter, setShowFooter] = useState(true);

    useEffect(() => {
        const handleScroll = () => {
            const scrollTop = window.scrollY;
            const windowHeight = window.innerHeight;
            const documentHeight = document.documentElement.scrollHeight;

            // Hiện footer nếu đang ở đầu hoặc cuối trang
            if (scrollTop < 50 || scrollTop + windowHeight >= documentHeight - 50) {
                setShowFooter(true);
            } else {
                setShowFooter(false);
            }
        };

        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    if (!showFooter) return null;

    return (
        <footer>
            <div className="Container fade-up">
                <div>
                    <p>SWminh0918195615@gmail.com</p>
                </div>
                <ul>
                    <li><a href="tel:0934190061"><FontAwesomeIcon icon={faMobileScreenButton} /> 0934.1900.61</a></li>
                    <li><a href=""><FontAwesomeIcon icon={faFacebookF} /> Facebook</a></li>
                    <li><a href=""><FontAwesomeIcon icon={faFacebookMessenger} /> Messenger</a></li>
                    <li><a href=""><FontAwesomeIcon icon={faInstagram} /> Instagram</a></li>
                    <li><a href=""><FontAwesomeIcon icon={faTelegram} /> Telegram</a></li>
                </ul>
            </div>
        </footer>
    );
};

export default Footer;
