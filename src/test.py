import cv2
import numpy as np

filepath = "../resources/image/000000.png"
image = cv2.imread(filepath, 1)
data = np.array([image[i, 2 * i] for i in range(10)]).T
print(data, type(data))
print(np.size(data, 0))
data[0, :] = data[0, :] / data[2, :]
print(data[0])
# cv2.imshow("image", self.image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


