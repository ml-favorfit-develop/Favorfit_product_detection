from PIL import Image
import os
import shutil

base_dir = "/media/mlfavorfit/sda/mask"

def copy_image(source_path, destination_path):
    shutil.copy(source_path, destination_path)
    return

def copy_and_save_image(image_array, image_path, tag, file_name, ext):
    os.makedirs(os.path.join(base_dir, tag), exist_ok=True)
    destination_path = os.path.join(base_dir, tag)

    copy_image(image_path, destination_path)
    Image.fromarray(image_array).save(os.path.join(destination_path, file_name), format=ext)
    return


def save_log(text):
    with open("log.txt", mode="a", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")
    return
