## Trả lời đúng 10 câu hỏi mục 6.1 của PDF

1. **Forward và Viterbi khác phép toán nào?** Forward cộng qua mọi trạng thái trước đó: αₜ(j)=bⱼ(xₜ)Σᵢαₜ₋₁(i)aᵢⱼ, trả likelihood của toàn bộ chuỗi. Viterbi thay Σ bằng max, lưu backpointer và truy ngược, trả đường trạng thái có xác suất lớn nhất. Trong log domain, tổng xác suất dùng logsumexp, còn chọn đường dùng max.

2. **γₜ(i) và ξₜ(i,j) là trách nhiệm gì?** γₜ(i)=P(qₜ=i|X,λ) là phần trách nhiệm state i nhận đối với frame t. ξₜ(i,j)=P(qₜ=i,qₜ₊₁=j|X,λ) là trách nhiệm của chuyển i→j tại thời điểm t. EM cộng các trách nhiệm này làm số đếm mềm để cập nhật Gaussian và A.

3. **Vì sao Baum–Welch có thể hội tụ xấu?** EM chỉ đảm bảo hướng tăng mục tiêu khi các điều kiện ước lượng phù hợp, không bảo đảm cực đại toàn cục. Khởi tạo không phù hợp, ít utterance nhưng quá nhiều state/tham số, covariance suy biến, nhãn sai hoặc silence kéo dài đều có thể làm nghiệm xấu. Cần xem history/cảnh báo, thử seed/khởi tạo trên train và chọn cấu hình bằng validation, không chọn trên test. Regularization và sai số số học có thể làm history giảm nhẹ.

4. **Vì sao ưu tiên covariance diag?** Với D=39, diag có 39 phương sai/state, full có D(D+1)/2=780 phần tử covariance độc lập/state. Full cần nhiều quan sát độc lập để ước lượng tương quan và đảo ma trận ổn định. Năm state cần 3900 tham số covariance full so với 195 diag, chưa tính 195 tham số mean; 6 utterance/lớp rất ít dù có nhiều frame tương quan.

5. **Vì sao delta vẫn hữu ích?** HMM mô hình hóa diễn tiến state nhưng giả định quan sát độc lập có điều kiện theo state. Delta và delta² đưa thông tin biến đổi lân cận vào một vector phát xạ, bổ sung hướng và độ cong của quỹ đạo phổ. Chúng cũng làm các chiều/frame tương quan hơn, nên không bảo đảm luôn tăng accuracy.

6. **Khi nào whole-word không cần lexicon?** Khi mỗi nhãn trong từ vựng nhỏ đã có model whole-word riêng, có thể score trực tiếp từng model. Khi dùng chung phone/subword để ghép thành các từ mới, cần lexicon hoặc cơ chế ánh xạ phát âm tương đương để xác định chuỗi đơn vị. Phải dùng inventory thống nhất, không đồng nhất chữ viết với phone.

7. **Self-loop giúp gì khi nói chậm?** Self-loop cho phép ở một state nhiều frame hơn, kéo dài đoạn âm học mà không tăng số state. Với xác suất self-loop aᵢᵢ cố định, thời lượng state không cuối có dạng hình học, kỳ vọng 1/(1−aᵢᵢ) frame. State cuối absorbing có thời lượng bị chặn bởi kết thúc utterance, không áp dụng kỳ vọng hữu hạn này. Biến đổi tốc độ quá lớn vẫn có thể gây mismatch phát xạ.

8. **Có cần chia log likelihood cho T để chọn model?** Không, trong một utterance mọi model score cùng X và T>0 như nhau: argmax s = argmax s/T. Chia T hữu ích để so giữa các utterance khác thời lượng hoặc đề xuất reject. Khi cộng LM, chia acoustic cho T thay đổi tương quan acoustic/LM nên phải nêu rõ thang điểm.

9. **DTW warping path so với Viterbi state path?** DTW căn hai chuỗi quan sát trong lưới (frame test, frame template), tối thiểu tổng chi phí cục bộ. Viterbi căn frame với state ẩn trong trellis (time, state), tối đa xác suất joint gồm transition và emission của model học từ nhiều mẫu. Cả hai dùng quy hoạch động và truy vết, nhưng trục thứ hai và ý nghĩa score khác nhau.

10. **Vì sao OOV vẫn bị chọn lớp và reject ra sao?** Argmax trên tập 5 model không có nhãn unknown nên luôn trả một nhãn, kể cả khi tất cả score thấp. Có thể reject nếu max score/frame < τ hoặc gap top1-top2/frame < δ, hoặc bổ sung model nền. τ và δ phải hiệu chuẩn bằng validation có in-vocabulary và OOV độc lập. Chưa có OOV trong bộ dữ liệu này nên chưa kết luận false accept/false reject hay hiệu quả unknown.
