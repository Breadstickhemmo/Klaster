# server/test.py

import zipfile
import os
import io

# --- Конфигурация ---

# Путь к основному (внешнему) zip-архиву
outer_zip_path = os.path.join('uploads', 'a65b4c25-8a2e-4cfe-84a9-e55540025438_Telegram.zip')

# Путь к вложенному zip-архиву ВНУТРИ внешнего архива
# Используем '/' как разделитель пути внутри zip
inner_zip_path_in_outer = 'Telegram/@doctorwine.zip'

# Путь к целевому файлу ВНУТРИ вложенного архива
# Используем '/' как разделитель пути внутри zip
target_file_path_in_inner = '@doctorwine/photos/photo_1258@03-12-2022_12-44-28.jpg'

# Папка для сохранения извлеченного файла
output_dir = 'contact_sheets'

# Имя файла для сохранения (можно взять из target_file_path_in_inner)
output_filename = os.path.basename(target_file_path_in_inner)
output_file_path = os.path.join(output_dir, output_filename)

# --- Логика извлечения ---

print(f"Попытка извлечь '{target_file_path_in_inner}'")
print(f"из '{inner_zip_path_in_outer}' внутри '{outer_zip_path}'")
print(f"в '{output_file_path}'")

try:
    # 1. Открыть внешний архив
    if not os.path.exists(outer_zip_path):
        print(f"Ошибка: Внешний архив не найден: {outer_zip_path}")
        exit()

    with zipfile.ZipFile(outer_zip_path, 'r') as outer_zip:
        print(f"Открыт внешний архив: {outer_zip_path}")

        # 2. Прочитать вложенный архив в память
        try:
            inner_zip_bytes = outer_zip.read(inner_zip_path_in_outer)
            print(f"Найден и прочитан вложенный архив: {inner_zip_path_in_outer}")
        except KeyError:
            print(f"Ошибка: Вложенный архив '{inner_zip_path_in_outer}' не найден внутри '{outer_zip_path}'")
            print("Содержимое внешнего архива:")
            outer_zip.printdir()
            exit()

        # 3. Открыть вложенный архив из памяти
        with zipfile.ZipFile(io.BytesIO(inner_zip_bytes), 'r') as inner_zip:
            print(f"Открыт вложенный архив из памяти.")

            # 4. Прочитать целевой файл из вложенного архива
            try:
                target_file_bytes = inner_zip.read(target_file_path_in_inner)
                print(f"Найден и прочитан целевой файл: {target_file_path_in_inner}")
            except KeyError:
                print(f"Ошибка: Целевой файл '{target_file_path_in_inner}' не найден внутри вложенного архива.")
                print("Содержимое вложенного архива:")
                inner_zip.printdir()
                exit()

    # 5. Создать папку назначения, если ее нет
    os.makedirs(output_dir, exist_ok=True)
    print(f"Папка назначения '{output_dir}' проверена/создана.")

    # 6. Сохранить извлеченный файл
    with open(output_file_path, 'wb') as outfile:
        outfile.write(target_file_bytes)
    print(f"Файл успешно сохранен как: {output_file_path}")

except zipfile.BadZipFile:
    print(f"Ошибка: Один из архивов поврежден.")
except FileNotFoundError:
     print(f"Ошибка: Не удалось найти файл или директорию при попытке сохранения.")
except Exception as e:
    print(f"Произошла непредвиденная ошибка: {e}")