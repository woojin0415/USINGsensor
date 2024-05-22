from django.urls import path
from usingsensor.views import *
urlpatterns = [
path("usingsensor/", detection),
    path('delete/', delete)

]

