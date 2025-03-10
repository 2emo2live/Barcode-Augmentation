import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import random


class ImageEvaluationApp:
    def __init__(self, root, image_folder):
        self.root = root
        self.root.title("Image Evaluation Tool")

        self.image_folder = image_folder
        self.images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if
                       img.endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(self.images)
        self.current_image_index = 0

        self.fidelity_score = tk.IntVar()

        self.load_image()
        self.create_widgets()

    def load_image(self):
        if self.current_image_index < len(self.images):
            image_path = self.images[self.current_image_index]
            self.image = Image.open(image_path)
            self.image.thumbnail((400, 400))
            self.photo = ImageTk.PhotoImage(self.image)
        else:
            messagebox.showinfo("Оценка качества завершена", "Все изображения оценены.")
            self.root.quit()

    def create_widgets(self):
        self.image_label = tk.Label(self.root, image=self.photo)
        self.image_label.pack(pady=10)

        fidelity_frame = tk.LabelFrame(self.root, text="Правдоподобность: это изображение реальное или искусствевнное?")
        fidelity_frame.pack(pady=10)

        tk.Radiobutton(fidelity_frame, text="1: Искусственное", variable=self.fidelity_score, value=1).pack(anchor=tk.W)
        tk.Radiobutton(fidelity_frame, text="2: Вероятно искусственное", variable=self.fidelity_score, value=2).pack(
            anchor=tk.W)
        tk.Radiobutton(fidelity_frame, text="3: Сложно ответить", variable=self.fidelity_score, value=3).pack(anchor=tk.W)
        tk.Radiobutton(fidelity_frame, text="4: Вероятно реальное", variable=self.fidelity_score, value=4).pack(anchor=tk.W)
        tk.Radiobutton(fidelity_frame, text="5: Реальное", variable=self.fidelity_score, value=5).pack(anchor=tk.W)

        next_button = tk.Button(self.root, text="Next", command=self.next_image)
        next_button.pack(pady=20)

    def next_image(self):
        fidelity = self.fidelity_score.get()
        print(f"Image {self.images[self.current_image_index]} - Fidelity: {fidelity}")

        self.fidelity_score.set(0)

        self.current_image_index += 1
        self.load_image()
        self.image_label.config(image=self.photo)


if __name__ == "__main__":
    root = tk.Tk()
    image_folder = "C:/Users/phone/2D_bar_codes/model/for_eval"
    app = ImageEvaluationApp(root, image_folder)
    root.mainloop()
