# MonReader

### <b>Background</b>

Our company develops innovative Artificial Intelligence and Computer Vision solutions that revolutionize industries. Machines that can see: We pack our solutions in small yet intelligent devices that can be easily integrated to your existing data flow. Computer vision for everyone: Our devices can recognize faces, estimate age and gender, classify clothing types and colors, identify everyday objects and detect motion. Technical consultancy: We help you identify use cases of artificial intelligence and computer vision in your industry. Artificial intelligence is the technology of today, not the future.

MonReader is a new mobile document digitization experience for the blind, for researchers and for everyone else in need for fully automatic, highly fast and high-quality document scanning in bulk. It is composed of a mobile app and all the user needs to do is flip pages and everything is handled by MonReader: it detects page flips from low-resolution camera preview and takes a high-resolution picture of the document, recognizing its corners and crops it accordingly, and it dewarps the cropped document to obtain a bird's eye view, sharpens the contrast between the text and the background and finally recognizes the text with formatting kept intact, being further corrected by MonReader's ML powered redactor.

<img src="https://go.apziva.com/static/img/project_10_1.png"><br>

<img src="https://go.apziva.com/static/img/project_10_2.jpg"><br>

### <b>Data Description</b>

We collected page flipping video from smart phones and labelled them as flipping and not flipping.

We clipped the videos as short videos and labelled them as flipping or not flipping. The extracted frames are then saved to disk in a sequential order with the following naming structure: VideoID_FrameNumber

### <b>Goal</b>
- Predict if the page is being flipped using a single image.

### <b> Success Metric</b>
- Evaluate model performance based on F1 score, the higher the better.

### <b>Other Objective</b>
- Predict if a given sequence of images contains an action of flipping.

### <b> Results</b>

<u>Model Performance</u>

We designed a convolutional neural network (CNN) specifically for a binary classification task, aiming to detect page-flipping actions. Our approach involved extensive experimentation and analysis, exploring a range of input shapes, filter counts, kernel sizes, pooling strategies, and more. These efforts led to an outstanding f1 score of 0.9832 on the test dataset. Notably, our model had the advantage of being trained on a substantial volume of high-resolution image data, comprising over two thousand video frames. This rich dataset significantly contributed to enhancing the model's accuracy and robustness.

<u>Future Work</u>

While the current iteration of our model is sufficiently accurate for deployment in a production environment, further enhancements can be achieved by integrating more diverse image data. Additionally, the knowledge and insights gained from this project can be leveraged to extend the model's capabilities, enabling the detection of various objects or motions using video frames or any image data.

### <b>Notebook and Installation</b>

For more details, you may refer to <a href='https://github.com/henryhyunwookim/MonReader/blob/main/MonReader.ipynb'>this notebook</a> directly.

To run MonReader.ipynb locally, please clone or fork this repo and install the required packages by running the following command:

pip install -r requirements.txt

##### <i>* Associated with Apziva</i>
