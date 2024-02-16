from django.urls import path

from . import views


app_name = "inventory"
urlpatterns = [
    # ex: /polls/
    path("", views.index, name="index"),
    # ex: /polls/5/
    path("results", views.index, name="results"),
    path("spectra/<slug:specimen_code>", views.spectra, name="spectra"),
    path("upload_spectra/", views.upload_spectra, name="upload_spectra"),
    path("upload_catalog", views.upload_catalog, name="upload_catalog"),
    path("download", views.download, name="download"),
    path("download_spectrum/<int:spectrum_id>", views.download_spectrum, name="download_spectrum"),
    path("search/<slug:specimen_code>", views.specimen, name="specimen"),
    path("accounts/login/", views.login_view, name="login_view"),
    path("accounts/logout/", views.logout_view, name="logout_view"),

]
