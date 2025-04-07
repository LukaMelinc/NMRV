import json
import os

def convert_json_to_yolo(json_file, output_dir, image_width, image_height):
    with open(json_file, 'r') as f:
        data = json.load(f)

    for image_name, bboxes in data.items():
        yolo_annotations = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            x_center = (x_min + x_max) / 2.0 / image_width
            y_center = (y_min + y_max) / 2.0 / image_height
            bbox_width = (x_max - x_min) / image_width
            bbox_height = (y_max - y_min) / image_height
            yolo_annotations.append(f"0 {x_center} {y_center} {bbox_width} {bbox_height}")

        output_file = os.path.join(output_dir, os.path.splitext(image_name)[0] + '.txt')
        with open(output_file, 'w') as f:
            f.write("\n".join(yolo_annotations))

# Example usage
convert_json_to_yolo('/Users/lukamelinc/Desktop/Faks/NMRV/Seminar/NMRV_seminar/thermal_annotations.json', '/Users/lukamelinc/Desktop/Faks/NMRV/Seminar/NMRV_seminar/annotations', 2208, 1242)
