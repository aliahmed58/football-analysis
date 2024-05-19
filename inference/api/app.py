from inference.api.make_celery import flask_app

if __name__ == '__main__':
    flask_app.run(debug=True)