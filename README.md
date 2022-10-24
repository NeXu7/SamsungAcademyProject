# Cell Classifier

### About the purpose
This project was created within the framework of IT.School
Academy by Samsung and based on Sechenov University dataset of kidney cancer cell classification

### How does it work
1) Program takes the json file with cells boundaries created by QuPath
2) WSI scan has to be in "slides/" dir
3) Then model create cell wise prediction and create json file with objects to import in QuPath

### How to use:
to test the program type 
"streamlit run main.py"
when being in root dir