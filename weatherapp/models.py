from django.db import models

# Create your models here.


class Map(models.Model):
    title = models.CharField(max_length=150)
    map_path = models.CharField(max_length=150)
    timestamp = models.DateTimeField()

    def __str__(self):
        return self.title
