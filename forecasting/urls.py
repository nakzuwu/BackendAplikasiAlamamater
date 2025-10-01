from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('forecast/', views.forecast, name='forecast'),
    path('api/forecast/', views.api_forecast, name='api_forecast'), 
]
