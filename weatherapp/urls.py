from django.urls import path, include

from weatherapp import views

urlpatterns = [
    path("", views.index),
]
