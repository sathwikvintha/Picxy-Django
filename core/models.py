from django.db import models


class Image(models.Model):
    image_path = models.CharField(max_length=255, unique=True)
    image_blob = models.BinaryField(null=True, blank=True)  # Binary image data
    image_name = models.CharField(max_length=255)
    meta_title = models.CharField(max_length=255, blank=True, null=True)
    meta_description = models.TextField(blank=True, null=True)
    labels = models.JSONField(default=list)
    confidence_scores = models.JSONField(default=dict)

    def __str__(self):
        return self.image_name
