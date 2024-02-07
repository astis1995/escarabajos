from django.urls import path

from . import views


app_name = "inventory"
urlpatterns = [
    # ex: /polls/
    path("", views.index, name="index"),
    # ex: /polls/5/
    path("results", views.index, name="results"),
    path("upload", views.upload, name="upload"),
    path("download", views.download, name="download"),
    path("search/<int:specimen_code>", views.specimen, name="specimen")
]
