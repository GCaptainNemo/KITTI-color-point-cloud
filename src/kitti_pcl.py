# import numpy as np
# import pcl.pcl_visualization
# from pcl import pcl_visualization
# import pcl
# # lidar_path 指定一个kitti 数据的点云bin文件就行了
# points = np.fromfile("../resources/Lidar/000001.bin", dtype=np.float32).reshape(-1, 4)  # .astype(np.float16)
# # 这里对第四列进行赋值，它代表颜色值，根据你自己的需要赋值即可；
# print(np.size(points, 0))
# points[:, 3] = 3329330
# # PointCloud_PointXYZRGB 需要点云数据是N*4，分别表示x,y,z,RGB ,其中RGB 用一个整数表示颜色；
# # 先对整数取256的模，余数为Blue的值,然后将整数商数再取256的模，余数为Green的值，得出的商取256的模，余数即为Red的值
# # 最后的商即为alpha的值。
# color_cloud = pcl.PointCloud_PointXYZRGB(points)
# visual = pcl_visualization.CloudViewing()
# visual.ShowColorCloud(color_cloud, b'cloud')
# flag = True
# while flag:
#     flag != visual.WasStopped()


import numpy as np
import cv2
from mayavi import mlab
import mayavi
import pcl.pcl_visualization
from pcl import pcl_visualization

# mayavi 4.7.2, vtk-9.0.1

class openKittiFiles:
    # p2_array = ...
    # Tr_array = ...
    # r0_array = ...
    def __init__(self):
        ...

    def open_pointcloud(self, filepath):
        self.pointcloud = np.fromfile(filepath, dtype=np.float32, count=-1).reshape([-1, 4])
        self.x = self.pointcloud[:, 0]
        self.y = self.pointcloud[:, 1]
        self.z = self.pointcloud[:, 2]
        self.r = self.pointcloud[:, 3]

        # print(np.size(pointcloud), type(pointcloud))


    def open_image(self, filepath):
        self.image = cv2.imread(filepath, 1)
        print("open image finish")
        # cv2.imshow("image", self.image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

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

        if vals == 'color':
            # true color point cloud
            size = np.size(self.pointcloud, 0)
            print("point cloud size = ", size)
            xy_coordinate = np.matmul(self.transform,
                                      np.concatenate((np.array([self.x]),
                                                      np.array([self.y]),
                                                      np.array([self.z]),
                                                      np.ones((1, size))),
                                                     axis=0))

            xy_coordinate[0, :] = xy_coordinate[0, :] / xy_coordinate[2, :]
            xy_coordinate[1, :] = xy_coordinate[1, :] / xy_coordinate[2, :]
            # print(int(xy_coordinate[0, :]))
            row_bound = self.image.shape[0]
            column_bound = self.image.shape[1]
            for i in range(size):
                if self.x[i] >= 0:
                    row = round(xy_coordinate[1][i])
                    column = round(xy_coordinate[0][i])

                    if 0 <= row < row_bound and 0 <= column < column_bound:
                        colori = self.image[round(xy_coordinate[1][i]),
                                   round(xy_coordinate[0][i])]
                        # rgb = np.concatenate((rgb, colori), axis=0)
                        print("colori = ", colori)
                        self.pointcloud[i, 3] = \
                            1 * 256 ** 3 + colori[0] * 256 ** 2 + colori[1] * 256 + colori[2]
                    else:
                        self.pointcloud[i, 3] = 0

                        # rgb = np.concatenate((rgb, np.zeros((1, 3))), axis=0)
                else:
                    self.pointcloud[i, 3] = 0
                    # rgb = np.concatenate((rgb, np.zeros((1, 3))), axis=0)

            color_cloud = pcl.PointCloud_PointXYZRGB(self.pointcloud)
            visual = pcl_visualization.CloudViewing()
            visual.ShowColorCloud(color_cloud, b'cloud')
            flag = True
            while flag:
                flag != visual.WasStopped()
            # 一维向量都是列向量，np.size(self.x, 0)表示列的数目

        else:
            if vals == "height":
                colar = self.z
            elif vals == 'dist':
                colar = np.sqrt(self.x ** 2 + self.y ** 2)  # Map Distance from sensor



if __name__ == "__main__":
    DAO = openKittiFiles()
    DAO.open_calib("../resources/calib/000001.txt")
    DAO.open_pointcloud("../resources/Lidar/000001.bin")
    DAO.open_image("../resources/image/000001.png")
    DAO.paint(vals="color")
    print("transform = ", DAO.transform)






