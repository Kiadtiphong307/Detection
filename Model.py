import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import defaultdict
import torch

# โหลดโมเดล YOLOv8 ที่แม่นยำขึ้น
model = YOLO("yolov8n.pt")  # เปลี่ยนเป็นโมเดลที่เล็กลงเพื่อความเร็ว
model.to('cuda' if torch.cuda.is_available() else 'cpu')  # ใช้ GPU ถ้ามี

# ตั้งค่าการเชื่อมต่อกล้อง
rtsp_url = "rtsp://admin:csmju-12345@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
cap = cv2.VideoCapture(rtsp_url)

# ตั้งค่าความละเอียดให้ตรงกับกล้อง
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 25)  # ตั้งค่า FPS ตามกล้อง

# เพิ่มการตั้งค่าเพื่อปรับปรุงการถอดรหัส HEVC
cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)  # เพิ่ม buffer size เล็กน้อย
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '5'))  

# เพิ่มการตรวจสอบการเชื่อมต่อ
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# กำหนดเส้น Counting Line
line_position = 550  # ตำแหน่งเส้นแนวนอน
offset = 10  # ขอบเขตตรวจจับ
entry_count = 0  # จำนวนคนเข้า
exit_count = 0  # จำนวนคนออก
people_in_room = 0  # จำนวนคนในห้องน้ำ
light_status = False  # ไฟเริ่มต้นเป็นปิด
last_entry_time = None  # เวลาคนเข้า

# Dictionary เก็บข้อมูลตำแหน่งของวัตถุในเฟรมก่อนหน้า
tracked_objects = defaultdict(lambda: {"previous": None, "current": None})

# เพิ่มตัวแปรสำหรับการจัดการ frame
frame_skip = 2  # ข้ามทุก 2 เฟรม
frame_count = 0

# สร้างหน้าต่างแสดงผลด้วยขนาดที่กำหนด
cv2.namedWindow("Line Crossing Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Line Crossing Detection", 1920, 1080)


while cap.isOpened():
    for _ in range(2):  # ข้าม frame เพื่อลดปัญหา frame dropping
        ret = cap.grab()
        if not ret:
            break
    
    ret, frame = cap.retrieve()  # อ่าน frame ที่ต้องการ
    if not ret:
        print("Error reading frame")
        time.sleep(0.5)  # เพิ่มเวลารอเมื่อมีปัญหา
        continue

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    
    # ดึงความละเอียดของเฟรม
    height, width = frame.shape[:2]
    
    # ตรวจจับวัตถุด้วย YOLO
    results = model(frame, conf=0.6)
    
    for result in results:
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if cls == 0 and conf > 0.6:  # ใช้ Threshold สูงขึ้นเพื่อลด False Positive
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # อัปเดตตำแหน่งของคนใน Dictionary
                tracked_objects[i]["previous"] = tracked_objects[i]["current"]
                tracked_objects[i]["current"] = center_y

                # ตรวจจับทิศทางของการเข้า/ออก
                if tracked_objects[i]["previous"] is not None:
                    if tracked_objects[i]["previous"] < line_position and center_y >= line_position:
                        # คนเดินเข้า
                        entry_count += 1
                        people_in_room += 1
                        last_entry_time = time.time()
                        light_status = True  # เปิดไฟ
                        print("[🔵] คนเข้า | เปิดไฟ ✅")

                    elif tracked_objects[i]["previous"] >= line_position and center_y < line_position:
                        # คนเดินออก
                        exit_count += 1
                        if people_in_room > 0:
                            people_in_room -= 1
                        print("[🔴] คนออก")

                # วาดกรอบรอบคน
                color = (0, 255, 0) if center_y < line_position else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (center_x, center_y), 5, color, -1)

    # แสดงเส้นนับคน
    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (255, 0, 0), 2)

    # ปิดไฟเมื่อครบ 30 นาที
    if light_status and last_entry_time and (time.time() - last_entry_time > 1800):
        light_status = False
        print("[⚠️] ปิดไฟอัตโนมัติ (30 นาทีครบ) ❌")

    # ปิดไฟเมื่อไม่มีคน
    if people_in_room == 0 and light_status:
        light_status = False
        print("[⚠️] ไม่มีคนในห้องน้ำ | ปิดไฟ ❌")

    # แสดงข้อมูลและเส้น
    text1 = f"Entry: {entry_count} | Exit: {exit_count} | People: {people_in_room} | Light: {'ON' if light_status else 'OFF'}"
    cv2.putText(frame, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # แสดงผลในขนาดเต็ม
    cv2.imshow("Line Crossing Detection", frame)

    # กด 'q' เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
