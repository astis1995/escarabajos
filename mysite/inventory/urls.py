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
    path("search/<slug:specimen_code>", views.specimen, name="specimen"),
    path("accounts/login/", views.login_view, name="login_view"),
    path("accounts/logout/", views.logout_view, name="logout_view"),

]
