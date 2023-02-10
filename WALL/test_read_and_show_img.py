import cv2

img_path = 'test_puff.jpg'
img = cv2.imread(img_path)
print(f"img.shape = {img.shape}")

cv2.imshow('preview', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
