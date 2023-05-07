
from imutils.perspective import four_point_transform
import cv2
from PIL import Image
from pytesseract import pytesseract
import openai
import os

img = cv2.imread("newImage.png")
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur_effect = cv2.GaussianBlur(grey, (7,7), 0)
limit = cv2.threshold(blur_effect, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

c = cv2.findContours(limit, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
c = c[0] if len(c) == 2 else c[1]
c = sorted(c, key=cv2.contourArea, reverse=True)
display = None

for c in c:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        display = approx
        break

warped = four_point_transform(img, display.reshape(4, 2))

cv2.imwrite("rotated.png", warped)
print("saved image")

path_to_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

path_to_image = 'rotated.png'

pytesseract.tesseract_cmd = path_to_tesseract

img = Image.open(path_to_image)

text = pytesseract.image_to_string(img)

print(text)

openai.api_key = os.getenv("OPENAI_API_KEY")

def getText(promptRequest):

    something = openai.Completion.create(
        model="text-davinci-003",
        prompt=promptRequest,
        max_tokens=2048,
        temperature=0.8
    )
    newtext = something['choices'][0]['text']

    return newtext

questions = getText("Here is some text: "+text+". From all the text you just read, can you give me every question?")
print(questions)

answers = getText("Here are some questions:"+questions+". Can you answer these questions one by one?")
print(answers)






