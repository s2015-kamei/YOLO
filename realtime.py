from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import pyrealsense2 as rs

# カメラの設定
conf = rs.config()
# RGB
conf.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# 距離
conf.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipe = rs.pipeline()
profile = pipe.start(conf)

# Alignオブジェクト生成
align_to = rs.stream.color
align = rs.align(align_to)

model = YOLO("runs/detect/train16/weights/best.pt")

try:
    while True:
        # フレーム待ち
        frames = pipe.wait_for_frames()
        aligned_frames = align.process(frames)

        #RGB
        RGB_frame = aligned_frames.get_color_frame()
        RGB_image = np.asanyarray(RGB_frame.get_data())

        #depyh
        depth_frame = aligned_frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)

        results = model.predict(RGB_image, show=False, verbose=False) 

        results_number = len(results)
        results_images = []
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

        for i, result_image in enumerate(results_images):
            cv2.imshow(f"result{i}", result_image)
        cv2.waitKey(1)

finally:
    pipe.stop()
    cv2.destroyAllWindows()
        


    
