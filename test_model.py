import os
import cv2
from ultralytics import YOLO
from pathlib import Path


def test_model_with_cuda():
    model_path = "best.pt"
    test_images_dir = "dataset.v1i.yolov8/test/images"

    if not os.path.exists(model_path):
        print(f"Không tìm thấy file mô hình: {model_path}")
        return

    if not os.path.exists(test_images_dir):
        print(f"Không tìm thấy thư mục test images: {test_images_dir}")
        return

    try:
        model = YOLO(model_path)

        import torch
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"Sử dụng GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print("CUDA không khả dụng, sử dụng CPU")

    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(test_images_dir).glob(f"*{ext}"))
        image_files.extend(Path(test_images_dir).glob(f"*{ext.upper()}"))

    image_files = sorted(image_files)

    if not image_files:
        print(f"Không tìm thấy ảnh nào trong thư mục: {test_images_dir}")
        return

    selected_images = image_files[::20]

    try:
        for i, image_path in enumerate(selected_images):
            results = model.predict(
                source=str(image_path),
                device=device,
                verbose=False
            )

            for result in results:
                im_array = result.plot()

                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for j, box in enumerate(boxes):
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = model.names[class_id] if class_id in model.names else f"class_{class_id}"

                height, width = im_array.shape[:2]
                original_index = image_files.index(image_path) + 1
                info_text = f"Selected {i+1}/{len(selected_images)} (Original #{original_index}): {image_path.name}"
                cv2.putText(im_array, info_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                max_height = 800
                if height > max_height:
                    scale = max_height / height
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    im_array = cv2.resize(im_array, (new_width, new_height))

                cv2.imshow('YOLO Detection Demo', im_array)

                key = cv2.waitKey(1000) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Lỗi: {e}")
    finally:
        cv2.destroyAllWindows()


def check_requirements():
    required_packages = ['ultralytics', 'opencv-python', 'torch']
    missing_packages = []

    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'ultralytics':
                from ultralytics import YOLO
            elif package == 'torch':
                import torch
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"Thiếu: {' '.join(missing_packages)}")
        return False
    return True


if __name__ == "__main__":
    if not check_requirements():
        exit(1)

    test_model_with_cuda()
