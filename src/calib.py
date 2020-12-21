import numpy as np
import cv2
from mayavi import mlab

import mayavi
import pcl


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
        print("open finish")

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
            size = np.size(self.x, 0)
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
            rgb = np.array([[0, 0, 0]])
            for i in range(size):
                if self.x[i] >= 0:
                    row = round(xy_coordinate[1][i])
                    column = round(xy_coordinate[0][i])
                    if 0 <= row < row_bound and 0 <= column < column_bound:
                        colori = np.array([self.image[round(xy_coordinate[1][i]),
                                   round(xy_coordinate[0][i])]])
                        rgb = np.concatenate((rgb, colori), axis=0)
                    else:
                        rgb = np.concatenate((rgb, np.zeros((1, 3))), axis=0)
                else:
                    rgb = np.concatenate((rgb, np.zeros((1, 3))), axis=0)
            rgb = rgb[1:, :]

            # for i in range(size):
            #     print(rgb[i, :])

            rgba = np.concatenate((rgb, 255 * np.ones((size, 1))), axis=1)

            # 一维向量都是列向量，np.size(self.x, 0)表示列的数目
            pts = mlab.pipeline.scalar_scatter(self.x, self.y, self.z)  # plot the points
            pts.add_attribute(rgba, 'colors')  # assign the colors to each point
            pts.data.point_data.set_active_scalars('colors')
            g = mlab.pipeline.glyph(pts)
            g.glyph.glyph.scale_factor = 0.1  # set scaling for all the points
            g.glyph.scale_mode = 'data_scaling_off'  # make all the points same size
            mlab.show()
        else:
            if vals == "height":
                col = self.z
            elif vals == 'dist':
                col = np.sqrt(self.x ** 2 + self.y ** 2)  # Map Distance from sensor
            fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
            mlab.points3d(self.x, self.y, self.z,
                          col,  # Values used for Color
                          mode="point",
                          # 灰度图映射成伪彩色图
                          colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                          # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                          figure=fig,
                          )
            mlab.points3d(0, 0, 0, color=(1, 1, 1), mode="sphere", scale_factor=1)
            axes = np.array(
                [[20.0, 0.0, 0.0, 0.0], [0.0, 20.0, 0.0, 0.0], [0.0, 0.0, 20.0, 0.0]],
                dtype=np.float64,
            )
            mlab.plot3d(
                [0, axes[0, 0]],
                [0, axes[0, 1]],
                [0, axes[0, 2]],
                color=(1, 0, 0),
                tube_radius=None,
                figure=fig,
            )
            # y轴
            mlab.plot3d(
                [0, axes[1, 0]],
                [0, axes[1, 1]],
                [0, axes[1, 2]],
                color=(0, 1, 0),
                tube_radius=None,
                figure=fig,
            )
            # z轴
            mlab.plot3d(
                [0, axes[2, 0]],
                [0, axes[2, 1]],
                [0, axes[2, 2]],
                color=(0, 0, 1),
                tube_radius=None,
                figure=fig,
            )
            mlab.show()


if __name__ == "__main__":
    DAO = openKittiFiles()
    DAO.open_calib("../resources/calib/000001.txt")
    DAO.open_pointcloud("../resources/Lidar/000001.bin")
    DAO.open_image("../resources/image/000001.png")
    DAO.paint(vals="color")
    print("transform = ", DAO.transform)


