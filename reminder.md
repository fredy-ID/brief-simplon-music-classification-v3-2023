
```sh
rm db.sqlite3
rm -r app/migrations
rm -r app/__pycache__
python manage.py makemigrations app
python manage.py migrate

```

```sh
del db.sqlite3
rmdir /s /q "app/migrations"
rmdir /s /q "app/__pycache__"
python manage.py makemigrations app
python manage.py migrate
```

- Missions create ajouter id_client
- Intervention dans main
- Assignment dans Agent
