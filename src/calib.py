import numpy as np
import cv2
from mayavi import mlab


# mayavi 4.7.2, vtk-9.0.1

class openKittiFiles:
    # p2_array = ...
    # Tr_array = ...
    # r0_array = ...
    def __init__(self):
        ...

    def open_pointcloud(self, filepath):
        pointcloud = np.fromfile(filepath, dtype=np.float32, count=-1).reshape([-1, 4])
        # print(np.size(pointcloud), type(pointcloud))
        self.x = pointcloud[:, 0]  # x position of point
        self.y = pointcloud[:, 1]  # y position of point
        self.z = pointcloud[:, 2]  # z position of point
        self.r = pointcloud[:, 3]  # reflectance value of point

    def open_image(self, filepath):
        self.image = cv2.imread(filepath, 1)

        cv2.imshow("image", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def open_calib(self, filepath):
        with open(filepath, "rb") as f:
            line_lst = f.readlines()
            p2 = str(line_lst[2])
            r0 = str(line_lst[4])
            Tr = str(line_lst[5])
        p2.strip()
        p2_str_lst = p2.split(" ")[1:]
        p2_str_lst[-1] = p2_str_lst[-1][:-3]
        self.p2_array = np.array([float(l) for l in p2_str_lst])
        self.p2_array = np.resize(self.p2_array, (3, 4))
        r0.strip()
        r0_str_lst = r0.split(" ")[1:]
        r0_str_lst[-1] = r0_str_lst[-1][:-3]
        self.r0_array = np.array([float(l) for l in r0_str_lst])
        self.r0_array = np.resize(self.r0_array, (3, 3))
        self.r0_array = np.concatenate((self.r0_array, np.zeros((3, 1))), axis=1)
        self.r0_array = np.concatenate((self.r0_array, np.array([[0, 0, 0, 1]])), axis=0)
        print("r0 = ", self.r0_array)
        Tr.strip()
        Tr_str_lst = Tr.split(" ")[1:]
        Tr_str_lst[-1] = Tr_str_lst[-1][:-3]
        self.Tr_array = np.array([float(l) for l in Tr_str_lst])
        self.Tr_array = np.resize(self.Tr_array, (3, 4))
        self.Tr_array = np.concatenate((self.Tr_array,
                                        np.array([[0, 0, 0, 1]])),
                                       axis=0)
        print("Tr = ", self.Tr_array)
        self.transform = np.matmul(self.p2_array, \
                                   np.matmul(self.r0_array , \
                                   self.Tr_array))

    def paint(self, vals="height"):
        # degr = np.degrees(np.arctan(self.z / d))
        if vals == "height":
            col = self.z
        elif vals == 'dist':
            col = np.sqrt(self.x ** 2 + self.y ** 2)  # Map Distance from sensor
        elif vals == 'color':
            size = np.size(self.x, 0)
            xy_coordinate = np.matmul(self.transform,
                            np.concatenate((np.array([self.x]),
                                            np.array([self.y]),
                                            np.array([self.z]),
                                            np.ones((1, size))),
                                            axis=0))
            xy_coordinate[0, :] = round(xy_coordinate[0, :] / xy_coordinate[2, :])
            xy_coordinate[1, :] = round(xy_coordinate[1, :] / xy_coordinate[2, :])
            col = np.array(self.image[xy_coordinate[1][i], xy_coordinate[0][i]]\
                           for i in range(size)).T
            # 一维向量都是列向量，np.size(self.x, 0)表示列的数目

            print(xy_coordinate[1])
            ...

        fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
        mlab.points3d(self.x, self.y, self.z,
                      col,  # Values used for Color
                      mode="point",
                      colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                      # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                      figure=fig,
                      )
        mlab.show()


if __name__ == "__main__":
    DAO = openKittiFiles()
    DAO.open_calib("../resources/calib/000000.txt")
    DAO.open_pointcloud("../resources/Lidar/000000.bin")
    DAO.open_image("../resources/image/000000.png")
    DAO.paint("color")
    print("transform = ", DAO.transform)


