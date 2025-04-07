import os
from PIL import Image

def resize_and_pad_image(image_path, output_path, target_size):
    with Image.open(image_path) as img:
        # Calculate the new size while maintaining the aspect ratio
        img.thumbnail((target_size, target_size))
        # Create a new image with the target size and white background
        new_img = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        # Paste the resized image onto the new image
        new_img.paste(img, ((target_size - img.size[0]) // 2, (target_size - img.size[1]) // 2))
        # Save the new image
        new_img.save(output_path)

def resize_images_in_directory(input_dir, output_dir, target_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            resize_and_pad_image(input_path, output_path, target_size)

# Example usage
input_dir_path = '/Users/lukamelinc/Desktop/Faks/NMRV/Seminar/NMRV_seminar/RGB_images/train/images'
output_dir_path = '/Users/lukamelinc/Desktop/Faks/NMRV/Seminar/NMRV_seminar/RGB_images/train/images_resized'
target_size = 640
resize_images_in_directory(input_dir_path, output_dir_path, target_size)
