# Cell Classifier 

### About the purpose
This project was created within the framework of IT.School
Academy by Samsung and based on Sechenov University dataset of kidney cancer cell classification
(this project is the winner of the contest)

### How does it work
1) Program takes the json file with cells boundaries created by QuPath
2) WSI scan has to be in "slides/" dir
3) Then model create cell wise prediction and create json file with objects to import in QuPath

### How to use:
1) install all requirements
2) streamlit run main.py
3) to test use test_image.tif or any image in test_images folder
