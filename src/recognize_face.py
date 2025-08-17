import cv2
def draw(frame, x, y, width, height, name, distance):
    # Vẽ bounding box
    cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 0, 255), 2)

    # Vẽ tên
    cv2.putText(frame, name, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Vẽ khoảng cách
    cv2.putText(frame, f'{distance:.2f}', (x + 100, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)