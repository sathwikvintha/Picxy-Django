from django.urls import path
from core import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("", views.index, name="index"),
    path("fetch_by_filename/", views.fetch_by_filename, name="fetch_by_filename"),
    path("detect/", views.detect, name="detect"),
    path("list_images/", views.list_images, name="list_images"),
    path("process_all_images/", views.process_all_images, name="process_all_images"),
    path(
        "fetch_all_images/",
        views.fetch_all_images_from_postgres,
        name="fetch_all_images",
    ),
    path("upload_image/", views.upload_image, name="upload_image"),
    path("delete-image/<int:image_id>/", views.delete_image, name="delete_image"),
    path("search/", views.search_images, name="search_images"),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
