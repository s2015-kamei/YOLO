from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import pyrealsense2 as rs
import socket


class RS_YOLO:
    
    def __init__(self):
        # カメラの設定
        self.conf = rs.config()
        # RGB
        self.conf.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # 距離
        self.conf.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.pipe = rs.pipeline()
        self.profile = self.pipe.start(self.conf)

        # Alignオブジェクト生成
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        self.model = YOLO("runs/detect/train16/weights/best.pt")

    def get_distance(self):
        
        # フレーム待ち
        frames = self.pipe.wait_for_frames()
        aligned_frames = self.align.process(frames)

        #RGB
        RGB_frame = aligned_frames.get_color_frame()
        RGB_image = np.asanyarray(RGB_frame.get_data())

        #depyh
        depth_frame = aligned_frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)

        results = self.model.predict(RGB_image, show=False, verbose=False) 

        results_number = len(results)
        results_images = []
        results_data = []
        for result in results:
            #print(result.boxes.xyxy.size())
            if(result.boxes.xyxy.shape[0] == 0):
                continue
            for i in range(result.boxes.xyxy.shape[0]):

                image = Image.fromarray(RGB_image)
                image = image.crop((result.boxes.xyxy[i][0].item(), result.boxes.xyxy[i][1].item(), result.boxes.xyxy[i][2].item(), result.boxes.xyxy[i][3].item()))
                image = np.array(image, dtype=np.uint8)

                x_size = image.shape[1]
                y_size = image.shape[0]

                
                #　画像を中心を維持したまま正方形にする
                if x_size > y_size:
                    image = image[:, int((x_size - y_size) / 2):int((x_size + y_size) / 2)]
                else:
                    image = image[int((y_size - x_size) / 2):int((y_size + x_size) / 2), :]


                # 画像の中央50% * 50%の範囲の深度を取得
                x1 = int(result.boxes.xyxy[i][0].item()) + int(image.shape[0] / 4)
                x2 = int(result.boxes.xyxy[i][2].item()) - int(image.shape[0] / 4)
                y1 = int(result.boxes.xyxy[i][1].item()) + int(image.shape[1] / 4)
                y2 = int(result.boxes.xyxy[i][3].item()) - int(image.shape[1] / 4)

                depth_list = [depth_frame.get_distance(x, y) for x in range(x1, x2 + 1) for y in range(y1, y2 + 1)]
                depth_average = sum(depth_list) / len(depth_list)
                
                print(depth_average)

                # 表示
                results_images.append(image)
                results_data.append((x1, x2, y1, y2, depth_average))

        for i, result_image in enumerate(results_images):
            cv2.imshow(f"result{i}", result_image)
        cv2.waitKey(1)
        return results_data

        # finally:
        #     self.pipe.stop()
        #     cv2.destroyAllWindows()
        #     return None
        

if __name__ == "__main__":
    serv_address = ('127.0.0.1', 1357)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    yolo = RS_YOLO()
    while True:
        data = yolo.get_distance()
        if data is not None:
            send_len = sock.sendto(data, serv_address)
        
        