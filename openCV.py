import cv2
import matplotlib.pyplot as plt
import numpy as np
img=cv2.imread(r"img1.JPG",0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(img)
plt.title("original image")
plt.axis("off")
plt.show()

# resize the image by 4/5 ratio and display it
print(img.shape)
resized=cv2.resize(img,(450,600))
plt.figure()
plt.imshow(resized)
plt.title("resized image")
plt.axis("off")
plt.show()


# add a text to the original image, for example "dog", and display it
text=cv2.putText(
    img,
    "dog",
    (400,100),
    cv2.FONT_HERSHEY_COMPLEX,
    1,
    (255,0,0)
)
plt.figure()
plt.imshow(text)
plt.title("adding text to image")
plt.axis("off")
plt.show()

# apply binary threshold: make pixels above 50 white, below 50 black,
_,thresh=cv2.threshold(img,thresh=50,maxval=255,type=cv2.THRESH_BINARY)
plt.figure()
plt.imshow(thresh,cmap="gray")
plt.title("threshold image")
plt.axis("off")
plt.show()

# apply Gaussian blur to the original image and display it
gaussian=cv2.GaussianBlur(img,ksize=(3,3),sigmaX=5)
plt.figure()
plt.imshow(gaussian,cmap="gray")
plt.title("gaussian blured image")
plt.axis("off")
plt.show()

# apply Laplacian gradient to the original image and display it
laplacian=cv2.Laplacian(img,ddepth=cv2.CV_16S)
plt.figure()
plt.imshow(laplacian,cmap="gray")
plt.title("laplacian image")
plt.axis("off")
plt.show()

# plot the histogram of the original image
hist=cv2.calcHist([img],channels=[0],mask=None,histSize=[256],ranges=[0,256])
plt.figure()
plt.imshow(hist,cmap="gray")
plt.title("histogram image")
plt.axis("off")
plt.show()


