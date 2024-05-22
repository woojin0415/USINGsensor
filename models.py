from django.db import models
class user_action(models.Model):
    action = models.CharField(max_length=100, default="")
    decouple = models.CharField(max_length=100, default="")
class action_db(models.Model):
    action = models.CharField(max_length=1500, default="")
    x_data = models.CharField(max_length=1500, default="")
    y_data = models.CharField(max_length=1500, default="")
    z_data = models.CharField(max_length=1500, default="")