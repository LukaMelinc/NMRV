from PIL import Image
import os

def get_image_size(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Get the dimensions of the image
        width, height = img.size
        # Get the file size in bytes
        file_size = os.path.getsize(image_path)

        return width, height, file_size

def main():
    # Path to the image file
    image_path = '/Users/lukamelinc/Desktop/Faks/NMRV/Seminar/NMRV_seminar/RGB_images/test/lj3_0_069051.png'

    # Get the image size
    width, height, file_size = get_image_size(image_path)

    # Print the image size
    print(f"Image Dimensions: {width}x{height}")
    print(f"File Size: {file_size} bytes")

if __name__ == "__main__":
    main()
