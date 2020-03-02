# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models
from django.utils.translation import ugettext_lazy as _

from image_cropping import ImageCropField, ImageRatioField
from testapp.presentation import InvoicePresentation

from django.contrib.auth.models import User
from rest_framework import serializers, viewsets, routers
from django.utils.encoding import python_2_unicode_compatible

from django.db.models.signals import post_save
from django.dispatch import receiver


from django.core.validators import RegexValidator


class Profile(models.Model):
	user = models.OneToOneField(User, on_delete=models.CASCADE)
	bio = models.TextField(max_length=500,blank=True)
	location = models.CharField(max_length=30,blank=True)
	birth_date = models.DateField(null=True,blank=True)
	email_confirmed = models.BooleanField(default=False)

@receiver(post_save,sender=User)
def update_user_profile(sender,instance,created, **kwargs):
	if created:
		Profile.objects.create(user=instance)
	instance.profile.save()


# Create your models here.
class Autor(models.Model):
    name = models.CharField(max_length=200)

    class Meta:
        ordering = ('pk',)
        permissions = (
            ("view_author", "Can see available Authors"),
        )



class Employee(models.Model):
	eid = models.CharField(max_length=20)
	ename = models.CharField(max_length=100)
	eemail = models.EmailField()
	econtact = models.CharField(max_length=15)
	class Meta:
		db_table = "employee"

class MonthlyWeatherByCity(models.Model):
	month = models.IntegerField()
	boston_temp = models.DecimalField(max_digits=5, decimal_places=1)
	houston_temp = models.DecimalField(max_digits=5, decimal_places=1)

class UserSerializer(serializers.HyperlinkedModelSerializer):
	class Meta:
		model = User
		fields = ('urls','username','email','is_staff')

class UserViewSet(viewsets.ModelViewSet):
	queryset = User.objects.all()
	serializer_class = UserSerializer

class DijkstraModel(models.Model):
	reg1 = models.IntegerField()
	reg4 = models.IntegerField()

	class Meta:
		db_table = "regional2"
