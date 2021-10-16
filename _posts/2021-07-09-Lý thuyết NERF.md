---
title:  "Tìm hiểu về neural rendering và mô hình NERF"
permalink: /blogs/Lý thuyết NERF
excerpt: ""
header:
  overlay_image: /assets/images/maxresdefault.jpg
  teaser: /assets/images/maxresdefault.jpg
  caption: ""
  actions:
    - label: "More Info"
      url: "https://www.matthewtancik.com/nerf"
categories:
  - Neural Rendering
toc: true
related: true
---

# 1/ Giới thiệu chung:

Với rất nhiều ứng dụng trong các lĩnh vực, ngành đồ họa máy tính luôn được quan tâm và phát triển không ngừng. Trong nỗ lực cải thiện các vấn đề còn tồn tại của quy trình đồ họa truyền thống, machine learning đã được đưa vào nghiên cứu như một giải pháp tiềm năng. Sự kết hợp giữa đồ họa máy tính và machine learning đã cho ra đời một hướng nghiên cứu vô cùng hấp dẫn - neural rendering.  

Trong bài viết này, chúng ta sẽ tìm hiểu về những khái niệm cơ bản của neural rendering và một mô hình neural rendering tiêu biểu của bài toán novel view synthesis - NERF.

*Các bạn có thể xem qua bài viết hướng dẫn chạy NERF tại [đây](http://quoccuonglqd.github.io/quoccuonguit/blogs/Guideline%20NERF)*

# 2/ Bối cảnh nghiên cứu:

## 2.1. Neural rendering là gì?

Quy trình đồ họa truyền thống liên quan đến những tác vụ thiết kế, căn chỉnh thủ công đối với các mô hình. Những công việc này đòi hỏi rất nhiều thời gian và công sức để hoàn thành. Tuy nhiên đây vẫn luôn là lựa chọn tối ưu của người làm đồ họa. Bởi lẽ với quy trình này, desiner có thể kiểm soát hoàn toàn các yếu tố của một mô hình, từ ánh sáng, vị trí camera, bề mặt cho đến texture, hiệu ứng đổ bóng, v.v. Và hiển nhiên là chất lượng của mô hình thiết kế sẽ tỉ lệ thuận với năng lực của desiner. Sản phẩm của 1 designer chuyên nghiệp sẽ có chất lượng rất cao.

Để giảm thiểu công sức thiết kế của designer, các hướng tiếp cận dựa trên machine learning đã được đưa vào nghiên cứu. Trước hết phải kể đến những nghiên cứu về mạng GAN. Những mô hình GAN sẽ cho phép máy tính tạo ra hình ảnh đầu ra từ việc huấn luyện trên dữ liệu cho trước. Nhưng bù lại cho ưu điểm trên, mô hình GAN được chứng minh là chưa đủ hiệu quả để mô tả những yếu tố chi tiết của mô hình như hiệu ứng đổ bóng, motion, v.v.


|Traditional Computer Graphic   | Generative Model                   |
|-------------------------------|------------------------------------|
|Ưu điểm:					    |Ưu điểm:					 		 |
|  - Chất lượng đầu ra cao      |  - Hoàn toàn tự động     		     |
|							    |  - Thời gian render nhanh		     |
|Khuyết điểm:				    |Khuyết điểm:				 		 |
|  - Tốn nhiều công sức	 	    |  - Đòi hỏi nhiều dữ liệu huấn luyện|
|  - Thời gian render lâu       |  - Không mô phỏng được các yếu tố chi tiết|

Neural rendering là một hướng tiếp cận kết hợp điểm mạnh của 2 người tiền bối. Các mô hình ML giờ đây sẽ được "trang bị" thêm những thành phần để thể hiện các tính chất vật lý từ đồ họa.

## 2.2. Bài toán Novel View Synthesis:

Đây là bài toán dự đoán hình ảnh từ các góc nhìn bất kỳ bằng cách cung cấp hình ảnh từ một số góc độ làm dữ liệu huấn luyện.
<img src="../assets/images/Xây dựng mô hình 3D với NERF/Novel view synthesis.png" width="600" />

**Hình 1** Novel View Synthesis. [Source](https://justusthies.github.io/posts/neuralrenderingtutorial_cvpr/)

## 2.3. Các kỹ thuật render:

Rendering hay chính là quá trình chuyển đổi các thông số của mô hình đồ họa thành hình ảnh. Nhìn chung, các kỹ thuật render có thể phân thành 2 nhóm chính.  

* Rasterization: Thông số của mô hình đồ họa được biểu diễn bởi một tập đối tượng trung gian. Có thể kể đến một số loại đối tượng trung gian sẽ là các tam giác, đa giác, mesh hay voxel. Mỗi một đối tượng trung gian sẽ ảnh hưởng đến giá trị của một số pixel trong hình ảnh render.  
* Ray tracing: Mô hình hóa quá trình truyền của các tia sáng. Đơn vị nhỏ nhất được xem xét là các hạt vật lý trong môi trường

NERF là một mô hình neural rendering kết hợp kỹ thuật ray tracing với mô hình ML để giải quyết bài toán Novel View Synthesis.

<img src="../assets/images/Xây dựng mô hình 3D với NERF/Nerf.png" width="600"/>

**Hình 2** Bối cảnh nghiên cứu NERF.

# 3/ Nội dung lý thuyết:

## 3.1. Ray-tracing volume rendering:

Từ *volume* mang ý nghĩa rằng thể tích không gian được mô phỏng sẽ được giới hạn trở lại. Các hiện tượng hấp thụ, phản xạ tia sáng sẽ chỉ được xem xét trên các hạt vật lý thuộc vùng không gian quy định

<img src="../assets/images/Xây dựng mô hình 3D với NERF/volume rendering.jpg" width="600"/>

**Hình 3** Mô phỏng quá trình tia sáng truyền đi. [Source](htttps://slidetodoc.comfast-high-accuracy-volume-rendering-thesis-defense-may)

Trong mô hình mô phỏng, giả định rằng các hạt vật lý lấp đầy trong không gian và không bị chồng lấp. Trên 1 tiết diện có diện tích đáy là <img src="https://render.githubusercontent.com/render/math?math=\Large E">, chiều cao <img src="https://render.githubusercontent.com/render/math?math=\Large \delta s">, có <img src="https://render.githubusercontent.com/render/math?math=\Large m(x)"> các hạt vật lý hình cầu có bán kính <img src="https://render.githubusercontent.com/render/math?math=\Large r"> bằng nhau. Khi tia sáng chiếu tới vị trí tiết diện sẽ có 2 khả năng:
* Tia sáng bị chặn lại bởi các hạt
* Tia sáng đi qua được kẽ hở giữa các hạt.

Vậy tại vị trí của tiết diện, có 2 loại nguồn sáng khác nhau:
* Nguồn sáng từ background đi xuyên qua
* Nguồn sáng phản xạ của bản thân các hạt tại vị trí này  

Tại 1 vị trí <img src="https://render.githubusercontent.com/render/math?math=\Large x_0"> , cường độ ánh sáng được bổ sung bởi nguồn sáng phản xạ tại đây    
![\Large](https://latex.codecogs.com/svg.latex?\Large&space;\lim_{x-x_0\to0}\frac{I(x)-I(x_0)}{x-x_0}=c(x_0)\frac{m(x_0)r^{2}\pi}{E}=c(x_0)\sigma(x_0)) (1)

Trong đó <img src="https://render.githubusercontent.com/render/math?math=\Large c(x_0)"> là giá trị cường độ ánh sáng của các hạt tại vị trí <img src="https://render.githubusercontent.com/render/math?math=\Large x_0">, <img src="https://render.githubusercontent.com/render/math?math=\Large \sigma(x_0)"> là xác suất ánh sáng bắt nguồn từ các hạt ở vị trí này thay vì là ánh sáng từ background.

Đồng thời, nguồn sáng background <img src="https://render.githubusercontent.com/render/math?math=\Large I_0"> bị hấp thụ bởi các hạt:

![\Large](https://latex.codecogs.com/svg.latex?\Large&space;\lim_{x-x_0\to0}\frac{I(x)-I(x_0)}{x-x_0}=-I_{0}\sigma(x_0)) (2)

Tổng hợp 2 công thức ở trên, ta thu được:

![\Large](https://latex.codecogs.com/svg.latex?\Large&space;\lim_{x-x_0\to0}\frac{I(x)-I(x_0)}{x-x_0}=c(x_0)\sigma(x_0)-I_{0}\sigma(x_0)) (3)

Giải phương trình vi phân này sẽ được:

![\Large](https://latex.codecogs.com/svg.latex?\Large&space;I(s)=\int_{0}^{s}c(x)\sigma(x)T(x)\,dx+I_{0}T(0)) (4)

![\Large](https://latex.codecogs.com/svg.latex?\Large&space;T(x)=exp(-\int_{x}^{s}\sigma(t)\,dt)) (5)

Bởi vì không xét đến nguồn sáng của background, nên ta có giá trị <img src="https://render.githubusercontent.com/render/math?math=\Large I_0=0">:  

![\Large](https://latex.codecogs.com/svg.latex?\Large&space;I(s)=\int_{0}^{s}c(x)\sigma(x)T(x)\,dx) (6)

Theo lý thuyết volume rendering, để tính giá trị màu từ cường độ ánh sáng, ta thay <img src="https://render.githubusercontent.com/render/math?math=\Large \sigma(x)"> bằng <img src="https://render.githubusercontent.com/render/math?math=\Large 1-exp(-\sigma(x))">  

![\Large](https://latex.codecogs.com/svg.latex?\Large&space;i(s)=\int_{0}^{s}c(x)(1-exp(-\sigma(x)))T(x)\,dx) (7)

## 3.2. Lượng tử hóa:

Để tính hàm tích phân ở trên, phương pháp lượng tử hóa được sử dụng thông qua việc sample một số lượng hữu hạn các điểm trong vùng không gian được giới hạn.

Giả sử tia sáng đi qua vùng không gian cắt tại 1 đoạn thẳng <img src="https://render.githubusercontent.com/render/math?math=\Large AB">. Ta chia  thành N đoạn bằng nhau; trên mỗi đoạn sample 1 điểm ở vị trí bất kỳ. Hàm tích phân ở vế phải phương trình (7) được ước lượng bởi:

![\Large](https://latex.codecogs.com/svg.latex?\Large&space;i(s)=\sum_{j=1}^{N}c(j)(1-exp(-\sigma(j)\delta(j)))T(j)\,dx) (8)
  
![\Large](https://latex.codecogs.com/svg.latex?\Large&space;T(j)=exp(-\sum_{k=1}^{j-1}\sigma(j)\delta(j))) (9)
  
![\Large](https://latex.codecogs.com/svg.latex?\Large&space;\delta(j)=t_{j+1}-t_j) (10)  
là khoảng cách giữa 2 điểm liên tiếp

## 3.3. Mô hình NERF:

Đến đây, ý tưởng chính của NERF là sử dụng một mạng MLP để tính các giá trị <img src="https://render.githubusercontent.com/render/math?math=\Large c"> và <img src="https://render.githubusercontent.com/render/math?math=\Large \sigma"> ở trên. Input của mạng là 1 vector 5 chiều thể hiện vị trí và góc độ của một điểm sample. Mạng MLP sẽ ánh xạ input trên thành giá trị <img src="https://render.githubusercontent.com/render/math?math=\Large c"> và <img src="https://render.githubusercontent.com/render/math?math=\Large \sigma"> tương ứng. Những cặp output sẽ được tổng hợp theo công thức (8) để tính toán giá trị pixel được dự đoán tương ứng

<img src="../assets/images/Xây dựng mô hình 3D với NERF/mlp.jpg" width="600"/>

**Hình 4** Mạng MLP ánh xạ input vị trí và góc độ sang <img src="https://render.githubusercontent.com/render/math?math=\Large c"> và <img src="https://render.githubusercontent.com/render/math?math=\Large \sigma">. [Source](https://i.ytimg.com/vi/dPWLybp4LL0/maxresdefault.jpg)

Quá trình training được thực hiện thông qua thuật toán backpropagation. Hàm loss được tính toán thông qua giá trị dự đoán pixel và giá trị ground truth của pixel đó trong dataset.

Kích cỡ của dataset sẽ tương đương với số lượng pixel trong dataset. Số lần forward mạng MLP trong 1 epoch train sẽ bằng kích cỡ dataset nhân với N. Trong đó N là số lượng điểm sample trên 1 tia tương ứng 1 pixel. Giá trị N càng lớn, thời gian huấn luyện mô hình sẽ càng lâu, nhưng độ chính xác và chi tiết của hình ảnh cũng sẽ tăng tương ứng. 

**_Lời kết:_** Trong bài viết này, chúng ta đã cùng tìm hiểu về lý thuyết volume ray-tracing rendering cũng như mô hình neural rendering NERF. Hẹn gặp các bạn trong các bài viết sau.

# Tham khảo

- Mildenhall, B., Srinivasan, P. P., Ortiz-Cayon, R., Kalantari, N. K., Ramamoorthi, R., Ng, R., & Kar, A. (2019). Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. ACM Transactions on Graphics (TOG), 38(4), 1-14.
- Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2020, August). Nerf: Representing scenes as neural radiance fields for view synthesis. In European conference on computer vision (pp. 405-421). Springer, Cham.
- Kajiya, J. T., & Von Herzen, B. P. (1984). Ray tracing volume densities. ACM SIGGRAPH computer graphics, 18(3), 165-174.