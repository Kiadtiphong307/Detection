import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import defaultdict

# โหลดโมเดล YOLOv8 ที่แม่นยำขึ้น
model = YOLO("yolov8n.pt")  # สามารถใช้ "yolov8l.pt" ได้ถ้าต้องการความแม่นยำมากขึ้น

cap = cv2.VideoCapture("rtsp://admin:csmju-12345@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0")

# กำหนดเส้น Counting Line
line_position = 300  # ตำแหน่งเส้นแนวนอน
offset = 10  # ขอบเขตตรวจจับ
entry_count = 0  # จำนวนคนเข้า
exit_count = 0  # จำนวนคนออก
people_in_room = 0  # จำนวนคนในห้องน้ำ
light_status = False  # ไฟเริ่มต้นเป็นปิด
last_entry_time = None  # เวลาคนเข้า

# Dictionary เก็บข้อมูลตำแหน่งของวัตถุในเฟรมก่อนหน้า
tracked_objects = defaultdict(lambda: {"previous": None, "current": None})

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ตรวจจับวัตถุในเฟรมโดยใช้ YOLOv8
    results = model(frame)

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

    # แสดงข้อมูลบนวิดีโอ
    text = f"Entry: {entry_count} | Exit: {exit_count} | People: {people_in_room} | Light: {'ON' if light_status else 'OFF'}"
    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # แสดงผลลัพธ์
    frame_resized = cv2.resize(frame, (1280, 720))
    cv2.imshow("Line Crossing Detection", frame_resized)

    # กด 'q' เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
