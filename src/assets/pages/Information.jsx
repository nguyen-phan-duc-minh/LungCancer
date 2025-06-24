import React from "react";
import Header from '../components/Header';
import Footer from '../components/Footer';

const Information = () => {
    return (
        <div className="OtherPage">
            <Header />
            <div className="Information">
                <h1>Tổng quan về Ung thư phổi</h1>

                <div className="Content">

                    <div className="Contain">
                        <h2>Ung thư phổi là gì?</h2>
                        <p>Ung thư phổi là khi các tế bào trong phổi phát triển không kiểm soát và xâm lấn vào các mô xung quanh tạo thành khối u phổi ác tính, phát triển nhanh về kích thước và chèn ép các cơ quan xung quanh. Ung thư phổi là một căn bệnh nguy hiểm vì có thể lan rộng đến các bộ phận khác của cơ thể thông qua hệ bạch huyết hoặc máu.</p>
                    </div>

                    <div className="Contain_img">
                        <img src="src/assets/images/infor1.png" alt="Minh họa khối u phổi" />
                        <span><strong>Hình minh họa:</strong> khí quản và phổi, cho thấy khối u lớn ở thùy trên của phổi trái.</span>
                    </div>

                    <div className="Contain">
                        <h2>Ung thư phổi có 2 loại:</h2>
                        <ul>
                            <li>Ung thư phổi không tế bào nhỏ: Là loại ung thư phổi thường gặp chiếm khoảng 85% các trường hợp ung thư phổi. Tốc độ phát triển và di căn của các tế bào ung thư phổi không tế bào nhỏ chậm hơn so với tế bào ung thư phổi tế bào nhỏ. Nếu được phát hiện sớm và điều trị kịp thời, người bệnh có thể có hy vọng sống cao.</li>
                            <li>Ung thư phổi tế bào nhỏ: Là loại ung thư nguy hiểm nhất trong các dạng ung thư phổi. Các tế bào ung thư thường xuất hiện ở đường dẫn khí lớn sau đó xâm lấn sang các cơ quan khác. Ung thư phổi tế bào nhỏ có tốc độ xâm lấn nhanh, di căn sớm nên điều trị khó khăn hơn.</li>
                        </ul>
                    </div>

                    <div className="Contain">
                        <h2>Nguyên nhân ung thư phổi</h2>
                        <p>Khói thuốc lá: Là nguyên nhân gây ung thư phổi cao nhất, bao gồm cả người hút thuốc lá và người thường xuyên tiếp xúc với khói thuốc lá (hút thuốc lá thụ động). Vì khi hít phải khói thuốc lá, các tế bào trong phổi sẽ bị tổn thương. Lúc đầu, cơ thể có thể tự chữa lành những tổn thương, nhưng khi tiếp xúc quá nhiều khói thuốc lá, phổi sẽ dần mất đi khả năng tự chữa lành, từ đó phổi sẽ hoạt động bất thường, làm tăng khả năng hình thành khối u ác tính ở phổi.</p>
                    </div>

                    <div className="Contain_img">
                        <img src="src/assets/images/infor2.png" alt="Triệu chứng ung thư phổi" />
                        <span><strong>Hình minh họa:</strong> mô phổi bị tổn thương gây triệu chứng hô hấp.</span>
                    </div>

                    <div className="Contain">
                        <p>Môi trường làm việc: Người làm việc trong môi trường tiếp xúc với nhiều khói bụi, hóa chất độc hại có thể sẽ gây ra những tổn thương cho phổi như sẹo, xơ hóa nhu mô phổi, hen suyễn, phổi tắc nghẽn mãn tính hoặc ung thư.</p>
                        <p>Yếu tố di truyền: Nếu một thành viên trong gia đình bị ung thư phổi thì nguy cơ mắc ung thư phổi ở những thành viên khác trong gia đình sẽ tăng cao.</p>
                        <p>Tiền sử bệnh phổi: Người có tiền sử mắc các bệnh như bệnh lao, bệnh phổi tắc nghẽn mãn tính (COPD), viêm phế quản, khí phế thủng… thì nguy cơ mắc bệnh ung thư phổi sẽ cao hơn những người không mắc bệnh.</p>
                    </div>

                    <div className="Contain">
                        <h2>Dấu hiệu nhận biết ung thư phổi</h2>
                        <p>Ung thư phổi ở giai đoạn sớm thường diễn tiến âm thầm, không gây ra triệu chứng rõ rệt hoặc chỉ xuất hiện một số dấu hiệu nhẹ như ho khan kéo dài, ho có đờm lẫn máu, uống thuốc không hiệu quả. </p>
                        <p>Khi ung thư phổi bắt đầu có triệu chứng rõ ràng hơn, thường đã ở giai đoạn muộn vì có thể đã di căn đến hạch bạch huyết, xương, gan, não và tuyến thượng thận, triệu chứng tùy thuộc vào mức độ và vị trí di căn.</p>
                        <p>Ung thư phổi là một căn bệnh nguy hiểm và có nguy cơ di căn đến các cơ quan khác trong cơ thể, do đó việc phát hiện sớm ung thư phổi là vô cùng quan trọng, góp phần nâng cao khả năng điều trị bệnh. Hãy chú ý đến các dấu hiệu sau đây để kịp thời phát hiện và điều trị kịp thời:</p>
                        <ul>
                            <li>Ho kéo dài, đôi khi ho có đờm hoặc máu.</li>
                            <li>Đau ngực trầm trọng hơn khi thở sâu hoặc ho.</li>
                            <li>Khàn tiếng.</li>
                            <li>Hụt hơi.</li>
                            <li>Thở khò khè.</li>
                            <li>Suy nhược, mệt mỏi.</li>
                            <li>Chán ăn.</li>
                            <li>Sụt cân.</li>
                        </ul>
                        <p>Một số dấu hiệu khi ung thư phổi di căn sang các bộ phận khác của cơ thể có thể gồm:</p>
                        <ul>
                            <li>Đau xương.</li>
                            <li>Đau đầu.</li>
                            <li>Sụt cân không rõ nguyên nhân.</li>
                            <li>Mất cảm giác thèm ăn.</li>
                            <li>Sưng ở mặt hoặc cổ.</li>
                        </ul>
                    </div>

                    <div className="Contain">
                        <h2>Ung thư phổi có lây truyền không?</h2>
                        <p>Ung thư phổi không phải là bệnh truyền nhiễm, không lây lan từ người này sang người khác qua tiếp xúc trực tiếp, không khí hoặc đồ dùng cá nhân.</p>

                        <h2>Cách chẩn đoán ung thư phổi</h2>
                        <p>Chẩn đoán ung thư phổi là một quá trình phức tạp bao gồm nhiều bước. Các bước chẩn đoán có thể bao gồm:</p>

                        <h3>Khám lâm sàng:</h3>
                        <p>
                        Bác sĩ sẽ thăm khám sức khỏe tổng quát để tìm kiếm các dấu hiệu của ung thư và các bệnh lý đi kèm. Ngoài ra, bác sĩ sẽ kiểm tra tiền sử sức khỏe của người bệnh và tiền sử gia đình người bệnh xem đã từng có ai mắc bệnh ung thư hay không.
                        </p>

                        <h3>Xét nghiệm chẩn đoán:</h3>
                        <p>Để chẩn đoán ung thư phổi, bác sĩ có thể sử dụng các phương pháp sau:</p>
                        <ul>
                        <li><strong>X-quang ngực:</strong> Giúp phát hiện các khối u hoặc bất thường trong phổi. Hầu hết các khối u phổi xuất hiện trên phim X-quang dưới dạng khối u màu trắng xám. Chẩn đoán có giá trị gợi ý nghi ngờ khối u, tuy nhiên X-quang ngực không thể đưa ra chẩn đoán chính xác.</li>
                        </ul>
                    </div>  

                    <div className="Contain_img">
                        <img src="src/assets/images/infor3.png" alt="Triệu chứng ung thư phổi" />
                        <span><strong>Hình minh họa:</strong> mô phổi bị tổn thương gây triệu chứng hô hấp.</span>
                    </div>

                    <div className="Contain">
                        <ul>
                            <li>
                                <strong>CT scan:</strong> Là kỹ thuật chẩn đoán hình ảnh sử dụng tia X để tạo ra hình ảnh cắt lớp đa chiều của phổi nhằm chẩn đoán các bệnh lý phổi. Tầm soát ung thư phổi hàng năm bằng cách chụp CT liều thấp giúp phát hiện sớm các tổn thương hoặc khối u trong phổi, đặc biệt đối với người hút thuốc lá lâu năm, người có tiền sử gia đình mắc ung thư phổi hoặc tiếp xúc với bụi than, hóa chất độc hại.
                            </li>
                            <li>
                                <strong>Nội soi phế quản:</strong> Kỹ thuật sử dụng một ống soi mềm, linh hoạt có gắn camera và đèn LED nhỏ để quan sát trực tiếp bên trong phổi. Ống soi được đưa vào qua đường mũi hoặc miệng, di chuyển qua cổ họng, khí quản và vào các nhánh khí quản (phế quản và tiểu phế quản) của phổi. Nội soi phế quản nhằm xác định vị trí tổn thương trong phế quản và qua đó sinh thiết khối u.
                            </li>
                            <li>
                                <strong>Sinh thiết u phổi:</strong> Khi các hình ảnh có được và cần xác định rõ bản chất khối u. Sinh thiết phổi có thể được thực hiện bằng phương pháp kín hoặc mở. Phương pháp kín được thực hiện sinh thiết xuyên qua da hoặc qua khí quản, sinh thiết mở được thực hiện trong phòng mổ và người bệnh được gây mê toàn thân.
                            </li>
                            <li>
                                <strong>Xét nghiệm đờm:</strong> Xét nghiệm tìm tế bào ung thư trong đờm.
                            </li>
                        </ul>
                    </div>  

                    <div className="Contain">
                        <h2>Những ai có nguy cơ cao mắc ung thư phổi?</h2>
                        <p>Nên tầm soát ung thư phổi nếu bạn thuộc một trong các nhóm nguy cơ cao sau:</p>
                        <ul>
                            <li>
                                <strong>Hút thuốc lá:</strong> Hút thuốc lá là yếu tố nguy cơ hàng đầu gây ung thư phổi, hút thuốc lá làm tăng nguy cơ ung thư phổi từ 15–30 lần so với các loại ung thư khác.
                            </li>
                        </ul>
                    </div>

                     <div className="Contain_img">
                        <img src="src/assets/images/infor4.png" alt="Triệu chứng ung thư phổi" />
                        <span><strong>Hình minh họa:</strong> mô phổi bị tổn thương gây triệu chứng hô hấp.</span>
                    </div>

                     <div className="Contain">
                        <ul>
                            <li>
                                <strong>Người từ 50 tuổi trở lên:</strong> Đặc biệt là những người đã hút thuốc trong nhiều năm, hoặc đã bỏ thuốc trong vòng 15 năm qua.
                            </li>
                            <li>
                                <strong>Tiếp xúc với khói thuốc lá thụ động:</strong> Hít phải khói thuốc cũng làm tăng nguy cơ mắc ung thư phổi gần như người hút chủ động.
                            </li>
                            <li>
                                <strong>Tiếp xúc với khí radon:</strong> Một loại khí độc sinh ra từ đất đá, vật liệu xây dựng như đá granite, gạch, hoặc nguồn nước ô nhiễm.
                            </li>
                            <li>
                                <strong>Có tiền sử bệnh phổi:</strong> Như bệnh phổi tắc nghẽn mãn tính (COPD), lao phổi cũ, viêm phổi mạn tính.
                            </li>
                            <li>
                                <strong>Tiếp xúc với phóng xạ, tia xạ:</strong> Đặc biệt là từng xạ trị vùng ngực trong điều trị ung thư khác.
                            </li>
                            <li>
                                <strong>Sống trong môi trường ô nhiễm không khí:</strong> Không khí nhiều bụi mịn, khói thải công nghiệp, khí độc hại.
                            </li>
                            <li>
                                <strong>Có tiền sử gia đình mắc ung thư phổi:</strong> Yếu tố di truyền có thể làm tăng nguy cơ mắc bệnh.
                            </li>
                        </ul>
                    </div>

                    <div className="Contain">
                        <h2>Cách phòng ngừa ung thư phổi</h2>
                        <p>Không có bất kỳ phương pháp phòng ngừa ung thư phổi tuyệt đối, nhưng có thể giảm nguy cơ mắc bệnh khi:</p>
                        <ul>
                            <li>Không hút thuốc lá và tránh xa khói thuốc lá.</li>
                            <li>Tránh tiếp xúc trực tiếp với khói bụi, nên đeo khẩu trang thường xuyên nếu phải làm việc tại môi trường nhiều khói bụi.</li>
                            <li>Xây dựng chế độ ăn nhiều trái cây, rau củ.</li>
                            <li>Tập thể dục đều đặn.</li>
                            <li>Khám sức khỏe định kỳ 6 tháng – 1 năm/lần để phát hiện sớm những tổn thương ở phổi, từ đó bác sĩ sẽ đưa ra phương pháp điều trị phù hợp.</li>
                        </ul>
                    </div>

                    <div className="Contain_img">
                        <img src="src/assets/images/infor5.png" alt="Triệu chứng ung thư phổi" />
                        <span><strong>Hình minh họa:</strong> mô phổi bị tổn thương gây triệu chứng hô hấp.</span>
                    </div>

                    <div className="Contain">
                        <p>
                        Chủ động bảo vệ sức khỏe bằng việc tầm soát ung thư phổi ngay nếu bạn thuộc nhóm người có nguy cơ cao. 
                        Tầm soát là các phương pháp giúp phát hiện ung thư phổi ở giai đoạn sớm, ngay cả khi chưa có triệu chứng, 
                        từ đó nâng cao khả năng điều trị triệt để bệnh.
                        </p>

                        <h2>Phương pháp điều trị ung thư phổi</h2>
                        <p>
                        Lựa chọn phương pháp điều trị ung thư phụ thuộc vào nhiều yếu tố, bao gồm:
                        </p>
                        <ul>
                            <li>Loại ung thư</li>
                            <li>Giai đoạn ung thư</li>
                            <li>Sức khỏe tổng thể của người bệnh</li>
                            <li>Các bệnh lý đi kèm</li>
                            <li>Nguyện vọng của người bệnh và gia đình</li>
                        </ul>
                        <p>
                        Không có phương pháp điều trị ung thư nào là hoàn hảo. 
                        Mỗi phương pháp điều trị ung thư đều cần có sự thăm khám và tư vấn từ bác sĩ để đưa ra quyết định điều trị phù hợp.
                        </p>

                        <h3>Phẫu thuật</h3>
                        <p>
                        Phẫu thuật là phương pháp điều trị ung thư phổ biến nhất. 
                        Mục tiêu là loại bỏ hoàn toàn khối u hoặc phần lớn khối u ở mô phổi xung quanh. 
                        Thường áp dụng cho giai đoạn sớm khi chưa di căn. 
                        Nếu khối u lớn hoặc đã di căn, phẫu thuật có thể không khả thi và phải kết hợp điều trị khác.
                        </p>

                        <h3>Xạ trị</h3>
                        <p>
                        Xạ trị sử dụng tia X năng lượng cao để tiêu diệt tế bào ung thư. Có thể dùng để:
                        </p>
                        <ul>
                            <li>Thu nhỏ khối u</li>
                            <li>Tiêu diệt tế bào ung thư còn sót lại</li>
                            <li>Giảm đau, chống chèn ép, cầm máu trong giai đoạn cuối</li>
                        </ul>

                        <h3>Hoá trị</h3>
                        <p>
                        Hoá trị sử dụng thuốc để tiêu diệt tế bào ung thư. 
                        Có thể được dùng đơn độc hoặc kết hợp với phẫu thuật hoặc xạ trị. 
                        Dạng dùng có thể là uống, tiêm truyền, và phối hợp đa tác nhân.
                        </p>

                        <h3>Liệu pháp miễn dịch</h3>
                        <p>
                        Liệu pháp miễn dịch giúp hệ thống miễn dịch nhận diện và tiêu diệt tế bào ung thư, 
                        bằng cách ngăn chặn các cơ chế mà ung thư dùng để trốn tránh miễn dịch.
                        </p>

                        <h3>Liệu pháp nhắm mục tiêu</h3>
                        <p>
                        Sử dụng các thuốc nhắm vào gen hoặc protein đặc hiệu liên quan đến sự phát triển của ung thư. 
                        Ví dụ như ức chế protein tăng trưởng, ngăn hình thành mạch máu mới hoặc sửa chữa DNA tổn thương.
                        </p>

                        <h3>Chăm sóc hỗ trợ</h3>
                        <p>
                        Các phương pháp chăm sóc nhằm nâng cao chất lượng sống cho người bệnh ung thư phổi:
                        </p>
                        <ul>
                            <li><strong>Giảm đau:</strong> Dùng thuốc để kiểm soát cơn đau.</li>
                            <li><strong>Chăm sóc hô hấp:</strong> Hỗ trợ hô hấp, giúp bệnh nhân dễ thở hơn.</li>
                            <li><strong>Dinh dưỡng:</strong> Đảm bảo cung cấp đủ dưỡng chất cho cơ thể.</li>
                            <li><strong>Tâm lý:</strong> Hỗ trợ tinh thần cho bệnh nhân và người thân.</li>
                        </ul>
                    </div>

                    <div className="Contain_img">
                        <img src="src/assets/images/infor6.png" alt="Triệu chứng ung thư phổi" />
                        <span><strong>Hình minh họa:</strong> mô phổi bị tổn thương gây triệu chứng hô hấp.</span>
                    </div>
                </div>
            </div>
            <Footer />
        </div>
    );
};

export default Information;
