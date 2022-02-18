from django.db import models


class one(models.Model):
    equip_name = models.CharField(max_length=100)
    is_right = models.BooleanField(default=True)
    render_coordinates = models.TextField(blank=True)
    column_distance = models.IntegerField(default=0)
    row_distance = models.IntegerField(default=0)
    radius = models.IntegerField(default=0)

    def __str__(self):
        return self.equip_name


class equip(models.Model):
    equip_name = models.CharField(max_length=100)

    def __str__(self):
        return self.equip_name
