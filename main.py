import torch
import cv2
import json
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image 
import matplotlib.pyplot as plt 
import uuid
import pytesseract
import sys
import re
doc_model = ocr_predictor(pretrained=True)
# Model
model = torch.hub.load('ultralytics/yolov5','custom' ,r'best.pt') # or yolov5n - yolov5x6, custom

# Images
#img = 'license3.jpg'  # or file, Path, PIL, OpenCV, numpy, list
#img = Image.open(r"license.jpg")  
#img=cv2.imread(r"license2.jpg")
img=cv2.imread(sys.argv[1])
#width, height = img.size

#print(width,height)
# Inference
results = model(img)

# Results
results.print() 
results.show()
#print(results.pandas().xyxy)
crops = results.crop(save=True)
#print(results.pandas().xyxy[0].to_json(orient="records"))

json_out=results.pandas().xyxy[0].to_json(orient="records")

for item in json.loads(json_out):
	org_img=img=cv2.imread(sys.argv[1])
	cropped_image = org_img[int(item['ymin']):int(item['ymax']), int(item['xmin']):int(item['xmax'])]
	plt.imshow(cropped_image)
	
	
  
	id = uuid.uuid4()
	file='temp/contour'+str(id)+'.png'
	cv2.imwrite(file, cropped_image)
	doc = DocumentFile.from_images(file)

	result = doc_model(doc)
	json_output = result.export()

	image=cv2.imread(file)
		# Convert to gray
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Resize the image
	image = cv2.resize(image, None, fx=2, fy=2)
	
	text = pytesseract.image_to_string(image)
	if(item['name']=='sex'):		 	
		val=re.findall(r'^(?:m|M|male|Male|f|F|female|Female)$',text)
		print(val)
	
	print(item['name'],':',text)	
	# print(json_output)



