from django.db import models

# Create your models here.


class Map(models.Model):
    title = models.CharField(max_length=150)
    map = models.ImageField(upload_to='maps/')
    time = models.DateTimeField()

    def __str__(self):
        return self.title
