from django.db import models
from django.contrib.auth.models import User


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


class Event(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    start_date = models.DateTimeField()
    end_date = models.DateTimeField()

    def __str__(self):
        return self.name


class Photo(models.Model):
    title = models.CharField(max_length=255)
    description = models.TextField()
    rating = models.IntegerField(default=0)
    views = models.IntegerField(default=0)
    pid = models.CharField(max_length=255, unique=True)
    published_at = models.DateTimeField(null=True, blank=True)
    modified_at = models.DateTimeField(auto_now=True)
    location_name = models.CharField(max_length=255, null=True, blank=True)
    lat = models.FloatField(null=True, blank=True)
    lon = models.FloatField(null=True, blank=True)
    status = models.CharField(
        max_length=1, choices=[("P", "Published"), ("U", "Unpublished")], default="U"
    )
    creator = models.ForeignKey(User, on_delete=models.CASCADE, related_name="photos")
    assigned_to = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="assigned_photos",
    )

    def __str__(self):
        return self.title


class Tag(models.Model):
    name = models.CharField(max_length=255)
    is_approved = models.BooleanField(default=False)
    approved_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True
    )

    def __str__(self):
        return self.name


class StatusChange(models.Model):
    changed_by = models.ForeignKey(User, on_delete=models.CASCADE)
    photo = models.ForeignKey("Photo", on_delete=models.CASCADE)
    from_status = models.CharField(max_length=1)
    to_status = models.CharField(max_length=1)
    changed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.photo.title} - {self.from_status} -> {self.to_status}"


class OEeventPhotos(models.Model):
    event = models.ForeignKey("Event", on_delete=models.CASCADE)
    photo = models.ForeignKey("Photo", on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.event.name} - {self.photo.title}"
