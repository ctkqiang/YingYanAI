from tensorflow.keras.preprocessing.image import ImageDataGenerator


class YingYanAI:
    def __init__(self) -> None:
        self.train_datagen = ImageDataGenerator(rescale=1.0 / 255)
        self.train_generator = train_datagen.flow_from_directory(
            "data/train",
            target_size=(224, 224),
            batch_size=32,
            class_mode="categorical",
        )

    def run(self) -> None:
        print("Hello, World!")

    def train(self) -> None:
        pass


if __name__ == "__main__":
    yyi: YingYanAI = YingYanAI()
    yyi.run()
