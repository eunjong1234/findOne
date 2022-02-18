from django.urls import path
from .views import oneApiView, diffApiView, listApiView

urlpatterns = [
    path('one/', oneApiView),
    path('list/', listApiView),
    path('diff/', diffApiView),
]
