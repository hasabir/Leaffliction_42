#!/usr/bin/env python3

import os
import sys
from collections import Counter
import matplotlib.pyplot as plt


class Distribution:
    def __init__(self, directory_path):
        self.directory_path = directory_path

    def get_lables(self, image_list):

        image_type = [image.split("/")[2] for image in image_list]
        ext_count = len(image_list)
        counter = Counter(image_type)
        labels = []
        sizes = []
        counte = []

        for ext, count in counter.items():
            labels.append(ext)
            counte.append(count)
            sizes.append((count / ext_count) * 100)
        print(f"Labels: {labels}, Sizes: {sizes}, Counts: {counte}")
        return labels, sizes, counte


    def bar_chart(self, image_list):

        labels, sizes, count = self.get_lables(image_list)
        plt.figure(figsize=(10, 6))
        colors = plt.cm.tab20.colors[:len(labels)]
        plt.bar(labels, count, color=colors)
        plt.xlabel('Image File Types')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


    def pie_chart(self, image_list):

        labels, sizes, _ = self.get_lables(image_list)

        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        plt.title('Image File Type Distribution')
        plt.show()


    def fetch_images(self, directory_path, file_list):
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"The directory {directory_path} does not exist.")
        for item_name in os.listdir(directory_path):
            full_path = os.path.join(directory_path, item_name)
            if os.path.isfile(full_path):
                file_list.append(full_path)
            elif os.path.isdir(full_path):
                self.fetch_images(full_path, file_list)
        return file_list


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Distribution.py <directory_name>")
        sys.exit(1)
    dir_name = sys.argv[1]
    try:
        distribution = Distribution(dir_name)
        file_list = []
        file_list = distribution.fetch_images(dir_name, file_list)
        distribution.pie_chart(file_list)
        distribution.bar_chart(file_list)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
