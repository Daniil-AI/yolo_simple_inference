from cryptography.fernet import Fernet
from ultralytics import YOLO
import io
import torch


# Генерация ключа шифрования
def generate_encrypted(original_model_path: str, encrypted_model_path: str):
    key = Fernet.generate_key()
    cipher_suite = Fernet(key)

    # Чтение и шифрование содержимого файла
    with open(original_model_path, "rb") as file:
        model_data = file.read()
    encrypted_model_data = cipher_suite.encrypt(model_data)

    # Сохранение зашифрованных весов в файл
    with open(encrypted_model_path, "wb") as file:
        file.write(encrypted_model_data)

    # Выводим ключ шифрования
    print(f"Encryption key: {key.decode()}")


# Функция для расшифровки файла с весами
def decrypt_model(
    encrypted_model_path: str, dummy_model_path: str, key: str, device: str
) -> YOLO:
    cipher_suite = Fernet(key.encode())

    with open(encrypted_model_path, "rb") as encrypted_file:
        encrypted_data = encrypted_file.read()

    decrypted_data = cipher_suite.decrypt(encrypted_data)

    model = YOLO(dummy_model_path)
    state_dict = torch.load(io.BytesIO(decrypted_data), map_location=device)
    model.load_state_dict(state_dict)

    return model


if __name__ == "__main__":
    generate_encrypted(
        original_model_path="../metadata/multiclass_FHD_special.pt",
        encrypted_model_path="../metadata/multiclass_FHD_special_encrypted.pt",
    )

"5m1dYS3ab8ZUCkbbPJmyFIGIsBFuVv-qzgkQS900yaw="
