# GitHub Report Generator

Веб-приложение для генерации отчетов по коммитам пользователей в GitHub репозиториях за выбранный период. Включает аутентификацию пользователей и хранение истории запросов.

## Обзор Функциональности

*   Регистрация и вход пользователей.
*   Создание запроса на генерацию отчета (указание URL репозитория, email автора коммитов, диапазон дат).
*   Фоновая обработка запроса на сервере с использованием GitHub API.
*   Просмотр истории своих запросов с текущим статусом (В обработке, Завершен, Ошибка).
*   (В планах) Скачивание сгенерированного отчета в формате JSON.

## Технологический стек

*   **Фронтенд:** React, TypeScript, CSS
*   **Бэкенд:** Flask (Python)
*   **База данных:** SQLAlchemy ORM (поддерживает SQLite по умолчанию, PostgreSQL и др.)
*   **Миграции БД:** Flask-Migrate
*   **Аутентификация:** Flask-JWT-Extended (JWT токены), Flask-Bcrypt (хэширование паролей)
*   **Работа с API:** Requests (для GitHub API)

## Структура проекта

```
/
├── README.md                       # Документация проекта
├── requirements.txt                # Зависимости Python для сервера
├── .gitignore                      # Файлы и папки, игнорируемые Git
├── LICENSE                         # Лицензия проекта
├── client/                         # Фронтенд-приложение на React + TypeScript
│   ├── package.json                # Зависимости и скрипты Node.js
│   ├── tsconfig.json               # Конфигурация TypeScript
│   ├── public/                     # Статические файлы (index.html, иконки и др.)
│   └── src/                        # Исходный код фронтенда
│       ├── App.tsx                 # Главный компонент React
│       ├── types.ts                # Типы TypeScript
│       └── components/             # React-компоненты
│            ├── Header.tsx         # Компонент шапки сайта
│            ├── ReportForm.tsx     # Компонент формы для создания нового запроса на отчет
│            ├── ReportTable.tsx    # Компонент таблицы
│            └── AuthModal.tsx      # Компонент модального окна
├── server/                         # Бэкенд-приложение на Flask (Python)
│   ├── app.py                      # Точка входа приложения, настройка Flask
│   ├── config.py                   # Конфигурация приложения
│   ├── models.py                   # Определения моделей базы данных
│   ├── routes.py                   # Основные API маршруты
│   ├── auth_routes.py              # Маршруты аутентификации (регистрация, вход)
│   ├── utils.py                    # Вспомогательные функции
│   ├── reports.py                  # Логика генерации отчетов
│   ├── logging_config.py           # Настройка логирования
│   ├── migrations/                 # Миграции базы данных (Flask-Migrate)
│   └── instance/                   # Папка для файлов конфигурации и базы данных
```

## Предварительные требования
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Windows (Git Bash/PowerShell)

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

Установите зависимости Python

```bash
pip install -r requirements.txt
```

Создайте файл .env в папке server/
и добавьте в него следующие переменные:

```bash
touch .env
```

**Содержимое файла .env:**

Обязательно: Ваш персональный токен GitHub с правами на чтение репозиториев
`GITHUB_TOKEN=ghp_YOUR_GITHUB_PERSONAL_ACCESS_TOKEN`

Обязательно: Секретный ключ для подписи JWT токенов
Пример генерации в Python: 

```bash
python -c 'import secrets; print(secrets.token_hex(32))'
```

`JWT_SECRET_KEY=ваш_очень_секретный_ключ_для_jwt`

Обязательно: Секретный ключ Flask (для сессий, защиты от CSRF и т.д.)
Пример генерации в Python: 

```bash
python -c 'import secrets; print(secrets.token_hex(32))'
```

`FLASK_SECRET_KEY=ваш_другой_очень_секретный_ключ_для_flask`

**3. Настройка базы данных (Flask-Migrate):**

Если вы запускаете проект ВПЕРВЫЕ и папки migrations еще нет:

```bash
flask db init
flask db migrate -m "Initial migration."
flask db upgrade
```

Если в будущем вы измените модели (models.py), повторите:

```bash
flask db migrate -m "Краткое описание изменений"
flask db upgrade
```

**4. Настройка Фронтенда (Клиент):**

Перейдите в папку клиента (из корневой папки проекта)

```bash
cd ../client
```

Установите зависимости Node.js

```bash
npm install
```

## Запуск Приложения

**1. Запустите Бэкенд (Сервер):**

Откройте новый терминал

```bash
cd server
flask run
```

Сервер будет доступен по адресу http://127.0.0.1:5000 (или http://localhost:5000)

**2. Запустите Фронтенд (Клиент):**

Откройте новый терминал

```bash
cd client
npm start
```

Фронтенд будет доступен по адресу http://localhost:3000